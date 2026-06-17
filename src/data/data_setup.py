import logging
import random
from numbers import Number

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from util.distribute import local_rank

from .eval.audiocaps import AudioCapsData
from .eval.video_dataset import VGGSound
from .extracted_audio import ExtractedAudio
from .extracted_vgg import ExtractedVGG
from .mm_dataset import MultiModalDataset

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(
        f"Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}"
    )


def load_vgg_data(data_cfg: DictConfig) -> Dataset:
    dataset = ExtractedVGG(latent=data_cfg.latent)
    return dataset


def load_audio_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    dataset = ExtractedAudio(
        data_dim=cfg.data_dim,
        latent=data_cfg.latent,
    )
    return dataset


def setup_training_datasets(
    cfg: DictConfig,
) -> tuple[Dataset, DistributedSampler, DataLoader]:
    vgg_oversample_rate = cfg.vgg_oversample_rate
    if not isinstance(vgg_oversample_rate, Number):
        vgg_oversample_rate = vgg_oversample_rate[cfg.variant]

    match cfg.train_mode:
        case "example":
            video = load_vgg_data(cfg.data.Example_video)
            audio = load_audio_data(cfg, cfg.data.Example_audio)
            dataset = MultiModalDataset([video], [audio])
        case "smoke":
            audio = load_audio_data(cfg, cfg.data.SmokeAudio)
            dataset = MultiModalDataset([], [audio])
        case "base":
            # load the largest one first
            freesound = load_audio_data(cfg, cfg.data.FreeSound)
            vgg = load_vgg_data(cfg.data.ExtractedVGG_train)
            audiocaps = load_audio_data(cfg, cfg.data.AudioCaps_train)
            audioset_sl = load_audio_data(cfg, cfg.data.AudioSetSL)
            bbcsound = load_audio_data(cfg, cfg.data.BBCSound)
            clotho = load_audio_data(cfg, cfg.data.Clotho_train)
            dataset = MultiModalDataset(
                [vgg] * vgg_oversample_rate,
                [
                    audiocaps,
                    audioset_sl,
                    bbcsound,
                    freesound,
                    clotho,
                ],
            )

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    return dataset, sampler, loader


def setup_test_datasets(cfg):
    match cfg.train_mode:
        case "example":
            dataset = load_vgg_data(cfg.data.Example_video)
        case "smoke":
            dataset = load_audio_data(cfg, cfg.data.SmokeAudio)
        case "base":
            dataset = load_vgg_data(cfg.data.ExtractedVGG_test)

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return loader


def setup_eval_dataset(
    dataset_name: str, cfg: DictConfig
) -> tuple[Dataset, DataLoader]:
    match dataset_name:
        case "audiocaps_full":
            dataset = AudioCapsData(
                cfg.eval_data.AudioCaps_full.audio_path,
                cfg.eval_data.AudioCaps_full.csv_path,
            )
        case "audiocaps":
            dataset = AudioCapsData(
                cfg.eval_data.AudioCaps.audio_path, 
                cfg.eval_data.AudioCaps.csv_path,
            )
        case "ExtractedVGG_test":
            dataset = VGGSound(
                cfg.eval_data.VGGSound.video_path,
                cfg.eval_data.VGGSound.csv_path,
                duration_sec=cfg.duration_s,
            )
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        error_avoidance=True,
    )
    return dataset, loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def construct_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
    error_avoidance: bool = False,
) -> tuple[DistributedSampler, DataLoader]:
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    train_loader = DataLoader(
        dataset,
        batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=error_avoidance_collate if error_avoidance else None,
        # NOTE: keep default fork (NOT forkserver) because dataset holds h5py
        # handles which are C-level and not picklable. forkserver/spawn need
        # to pickle the dataset object → "h5py objects cannot be pickled".
        # latent/loader.py uses forkserver because its dataset opens audio/video
        # files lazily per-iter (no persistent handle).
    )
    return train_sampler, train_loader
