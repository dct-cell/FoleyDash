from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vgg_sound import VGGSound
from wav_dataset import WavTextClipsDataset
from src.data.data_setup import error_avoidance_collate
from util.distribute import local_rank

from os import PathLike


def video_loader(
    cfg: DictConfig,
    path_data: PathLike,
    path_tsv: PathLike,
    sample_rate: int,
    duration_sec: float,
    num_samples: int,
):
    dataset = VGGSound(
        path_data,
        tsv_path=path_tsv,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        audio_samples=num_samples,
        normalize_audio=True,
    )
    sampler = DistributedSampler(
        dataset,
        rank=local_rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        drop_last=False,
        collate_fn=error_avoidance_collate,
        multiprocessing_context="forkserver" if cfg.num_workers > 0 else None,
    )

    return loader


def audio_loader(
    cfg: DictConfig,
    path_data: PathLike,
    captions_tsv: PathLike,
    clips_tsv: PathLike,
    sample_rate: int,
    num_samples: int,
):
    dataset = WavTextClipsDataset(
        path_data,
        captions_tsv=captions_tsv,
        clips_tsv=clips_tsv,
        sample_rate=sample_rate,
        num_samples=num_samples,
        normalize_audio=True,
        reject_silent=True,
    )
    sampler = DistributedSampler(
        dataset,
        rank=local_rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        drop_last=False,
        collate_fn=error_avoidance_collate,
    )
    return loader
