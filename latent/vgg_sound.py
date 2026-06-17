import logging
import os
from os import PathLike
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torchcodec.decoders import AudioDecoder, VideoDecoder
# torio removed in torchaudio 2.9; using torchcodec (matched ABI for torch 2.9 cu130).

from util.distribute import is_rank0

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VGGSound(Dataset):
    def __init__(
        self,
        root: PathLike,
        *,
        tsv_path: PathLike = "sets/vgg3-train.tsv",
        sample_rate: int = 16_000,
        duration_sec: float = 8.0,
        audio_samples: int = None,
        normalize_audio: bool = False,
    ):
        self.root = Path(root)
        self.normalize_audio = normalize_audio
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 15ms
            assert abs(effective_duration - duration_sec) < 0.015, (
                f"audio_samples {audio_samples} does not match duration_sec {duration_sec}"
            )

        videos = sorted(os.listdir(self.root))
        videos = set([Path(v).stem for v in videos])  # remove extensions
        self.labels = {}
        self.videos = []
        missing_videos = []

        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep="\t", dtype={"id": str}).to_dict("records")
        for record in df_list:
            id = record["id"]
            label = record["label"]
            if id in videos:
                self.labels[id] = label
                self.videos.append(id)
            else:
                missing_videos.append(id)

        if is_rank0:
            log.info(f"{len(videos)} videos found in {root}")
            log.info(f"{len(self.videos)} videos found in {tsv_path}")
            log.info(f"{len(missing_videos)} videos missing in {root}")

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = audio_samples
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

        self.resampler = {}

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        label = self.labels[video_id]

        video_path = str(self.root / (video_id + ".mp4"))
        vdec = VideoDecoder(video_path, dimension_order="NCHW")

        if vdec.metadata.duration_seconds < self.duration_sec:
            raise RuntimeError(f"Video too short {video_id}")

        clip_timestamps = [i / _CLIP_FPS for i in range(self.clip_expected_length)]
        sync_timestamps = [i / _SYNC_FPS for i in range(self.sync_expected_length)]
        clip_chunk = vdec.get_frames_played_at(clip_timestamps).data
        sync_chunk = vdec.get_frames_played_at(sync_timestamps).data

        if clip_chunk is None or clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video bad {video_id}, expected {self.clip_expected_length}, "
                f"got {None if clip_chunk is None else clip_chunk.shape[0]}"
            )
        if sync_chunk is None or sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f"Sync video bad {video_id}, expected {self.sync_expected_length}, "
                f"got {None if sync_chunk is None else sync_chunk.shape[0]}"
            )

        # process audio
        adec = AudioDecoder(video_path)
        sample_rate = int(adec.metadata.sample_rate)
        audio_samples = adec.get_all_samples().data  # (C, N)
        audio_chunk = audio_samples.mean(dim=0)  # mono
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
            if abs_max <= 1e-6:
                raise RuntimeError(f"Audio is silent {video_id}")

        # resample
        if sample_rate == self.sample_rate:
            audio_chunk = audio_chunk
        else:
            if sample_rate not in self.resampler:
                # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
            audio_chunk = self.resampler[sample_rate](audio_chunk)

        if audio_chunk.shape[0] < self.expected_audio_length:
            raise RuntimeError(f"Audio too short {video_id}")
        audio_chunk = audio_chunk[: self.expected_audio_length]

        # truncate the video
        clip_chunk = clip_chunk[: self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video wrong length {video_id}, "
                f"expected {self.clip_expected_length}, "
                f"got {clip_chunk.shape[0]}"
            )
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[: self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(
                f"Sync video wrong length {video_id}, "
                f"expected {self.sync_expected_length}, "
                f"got {sync_chunk.shape[0]}"
            )
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            "id": video_id,
            "caption": label,
            "audio": audio_chunk,
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
        return len(self.labels)
