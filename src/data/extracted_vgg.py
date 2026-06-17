import logging
from os import PathLike
from pathlib import Path

import h5py
import torch
from torch.utils.data.dataset import Dataset

from util.distribute import is_rank0

log = logging.getLogger()


class ExtractedVGG(Dataset):
    def __init__(
        self,
        *,
        latent: PathLike,
    ):
        super().__init__()
        if is_rank0:
            log.info(f"Loading precomputed latent from {latent}")
        latent = Path(latent)
        f = h5py.File(latent)
        self.id = f["id"]
        self.label = f["label"]
        self.mean = f["mean"]
        self.std = f["std"]
        self.clip_features = f["clip_features"]
        self.sync_features = f["sync_features"]
        self.text_features = f["text_features"]

        if is_rank0:
            log.info(f"Loaded {len(self)} samples.")
            log.info(f"Loaded mean: {self.mean.shape}.")
            log.info(f"Loaded std: {self.std.shape}.")
            log.info(f"Loaded clip_features: {self.clip_features.shape}.")
            log.info(f"Loaded sync_features: {self.sync_features.shape}.")
            log.info(f"Loaded text_features: {self.text_features.shape}.")

        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        latents = torch.from_numpy(self.mean[()]).cuda()
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = {
            "id": self.id[idx].decode("utf-8"),
            "a_mean": torch.from_numpy(self.mean[idx]),
            "a_std": torch.from_numpy(self.std[idx]),
            "clip_features": torch.from_numpy(self.clip_features[idx]),
            "sync_features": torch.from_numpy(self.sync_features[idx]),
            "text_features": torch.from_numpy(self.text_features[idx]),
            "caption": self.label[idx].decode("utf-8"),
            "video_exist": self.video_exist,
            "text_exist": self.text_exist,
        }

        return data

    def __len__(self):
        return len(self.id)
