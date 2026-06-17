import logging
from pathlib import Path
import h5py
import torch
from torch.utils.data.dataset import Dataset
from util.distribute import is_rank0
from os import PathLike

log = logging.getLogger()



class ExtractedAudio(Dataset):
    def __init__(
        self,
        *,
        latent: PathLike,
        data_dim: dict[str, int],
    ):
        super().__init__()
        self.data_dim = data_dim
        if is_rank0:
            log.info(f"Loading precomputed latent from {latent}")
        latent = Path(latent)
        f = h5py.File(latent)
        self.id = f["id"]
        self.label = f["label"]
        self.mean = f["mean"]
        self.std = f["std"]
        self.text_features = f["text_features"]

        if is_rank0:
            log.info(f"Loaded {len(self)} samples from {latent}.")
            log.info(f"Loaded mean: {self.mean.shape}.")
            log.info(f"Loaded std: {self.std.shape}.")
            log.info(f"Loaded text features: {self.text_features.shape}.")

        self.fake_clip_features = torch.zeros(
            self.data_dim["clip_seq_len"], self.data_dim["clip_dim"]
        )
        self.fake_sync_features = torch.zeros(
            self.data_dim["sync_seq_len"], self.data_dim["sync_dim"]
        )
        self.video_exist = torch.tensor(0, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        latents = torch.from_numpy(self.mean[()]).cuda()
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = {
            "id": self.id[idx].decode("utf-8"),
            "a_mean": torch.from_numpy(self.mean[idx]),
            "a_std": torch.from_numpy(self.std[idx]),
            "clip_features": self.fake_clip_features,
            "sync_features": self.fake_sync_features,
            "text_features": torch.from_numpy(self.text_features[idx]),
            "caption": self.label[idx].decode("utf-8"),
            "video_exist": self.video_exist,
            "text_exist": self.text_exist,
        }
        return data

    def __len__(self):
        return len(self.id)
