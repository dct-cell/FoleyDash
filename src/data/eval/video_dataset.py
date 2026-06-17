import logging
import os
from os import PathLike
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torchcodec.decoders import VideoDecoder
# torio (torchaudio.io) was removed in torchaudio 2.9. Replaced with torchcodec
# VideoDecoder, which is the matched-ABI video reader for torch 2.9 cu130.
# We sample frames at requested timestamps to emulate torio's frame_rate=N output.

from util.distribute import is_rank0

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VideoDataset(Dataset):
    def __init__(
        self,
        video_root: PathLike,
        *,
        duration_sec: float = 8.0,
    ):
        self.video_root = Path(video_root)

        self.duration_sec = duration_sec

        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose(
            [
                v2.Resize(
                    (_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.sync_transform = v2.Compose(
            [
                v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(_SYNC_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # to be implemented by subclasses
        self.captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]

        video_path = str(self.video_root / (video_id + ".mp4"))
        decoder = VideoDecoder(video_path, dimension_order="NCHW")

        clip_timestamps = [i / _CLIP_FPS for i in range(self.clip_expected_length)]
        sync_timestamps = [i / _SYNC_FPS for i in range(self.sync_expected_length)]

        if decoder.metadata.duration_seconds < self.duration_sec:
            raise RuntimeError(
                f"Video too short {video_id}, "
                f"expected {self.duration_sec}, got {decoder.metadata.duration_seconds}"
            )

        clip_chunk = decoder.get_frames_played_at(clip_timestamps).data
        sync_chunk = decoder.get_frames_played_at(sync_timestamps).data

        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video wrong length {video_id}, "
                f"expected {self.clip_expected_length}, "
                f"got {clip_chunk.shape[0]}"
            )
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(
                f"Sync video wrong length {video_id}, "
                f"expected {self.sync_expected_length}, "
                f"got {sync_chunk.shape[0]}"
            )

        clip_chunk = self.clip_transform(clip_chunk)
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            "name": video_id,
            "caption": caption,
            "clip_video": clip_chunk,
            "sync_video": sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception:
            # log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def __len__(self):
        return len(self.captions)


class VGGSound(VideoDataset):
    def __init__(
        self,
        video_root: PathLike,
        csv_path: PathLike,
        *,
        duration_sec: float = 8.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if is_rank0:
            log.info(f"{len(videos)} videos found in {video_root}")
        self.captions = {}

        df = pd.read_csv(
            csv_path, header=None, names=["id", "sec", "caption", "split"]
        ).to_dict(orient="records")

        videos_no_found = []
        for row in df:
            if row["split"] == "test":
                start_sec = int(row["sec"])
                video_id = str(row["id"])
                # this is how our videos are named
                video_name = f"{video_id}_{start_sec:06d}"
                if video_name + ".mp4" not in videos:
                    videos_no_found.append(video_name)
                    continue

                self.captions[video_name] = row["caption"]

        if is_rank0:
            log.info(f"{len(videos)} videos found in {video_root}")
            log.info(f"{len(self.captions)} useable videos found")
            if videos_no_found:
                log.info(
                    f"{len(videos_no_found)} found in {csv_path} but not in {video_root}"
                )
                log.info(
                    "A small amount is expected, as not all videos are still available on YouTube"
                )

        self.videos = sorted(list(self.captions.keys()))
