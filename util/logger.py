"""
Dumps things to tensorboard and console
"""

import datetime
import logging
import math
import os
from collections import defaultdict
from os import PathLike
from pathlib import Path

import colorlog
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

from .config import get_rundir
from .distribute import local_rank, is_rank0
from .time_estimator import PartialTimeEstimator, TimeEstimator

log = logging.getLogger()
log.setLevel(logging.INFO)

COLOR_TIME = "cyan"
COLOR_RANK = "purple"
formatter = colorlog.ColoredFormatter(
    fmt=f"[%({COLOR_TIME})s%(asctime)s%(reset)s][%({COLOR_RANK})sr{local_rank}%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
log.addHandler(handler)

if is_rank0:
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = get_rundir() / f"train-{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)


def tensor_to_numpy(image: torch.Tensor):
    image_np = (image.numpy() * 255).astype("uint8")
    return image_np


def detach_to_cpu(x: torch.Tensor):
    return x.detach().cpu()


def fix_width_trunc(x: float):
    return "{:.9s}".format("{:0.9f}".format(x))


class TensorboardLogger:
    def __init__(
        self,
        exp_id: str,
        run_dir: PathLike,
        py_logger: logging.Logger,
        *,
        is_rank0: bool = False,
    ):
        self.exp_id = exp_id
        self.run_dir = Path(run_dir)
        self.py_log = py_logger
        if is_rank0:
            self.tb_log = SummaryWriter(run_dir)
        else:
            self.tb_log = None

        # log the SLURM job id if available
        job_id = os.environ.get("SLURM_JOB_ID", None)
        if job_id is not None:
            self.log_string("slurm_job_id", job_id)

        # used when logging metrics
        self.batch_timer: TimeEstimator = None
        self.data_timer: PartialTimeEstimator = None

        self.nan_count = defaultdict(int)

    def log_scalar(self, tag: str, x: float, it: int):
        if self.tb_log is None:
            return
        if math.isnan(x) and "grad_norm" not in tag:
            self.nan_count[tag] += 1
        else:
            self.nan_count[tag] = 0
        self.tb_log.add_scalar(tag, x, it)

    def log_metrics(
        self,
        prefix: str,
        metrics: dict[str, float],
        it: int,
        ignore_timer: bool = False,
    ):
        msg = f"{self.exp_id}-{prefix} - it {it:6d}:"
        metrics_msg = ""
        for k, v in sorted(metrics.items()):
            self.log_scalar(f"{prefix}/{k}", v, it)
            metrics_msg += f"{k}:{v:.6f} "

        if self.batch_timer is not None and not ignore_timer:
            self.batch_timer.update()
            avg_time = self.batch_timer.get_and_reset_avg_time()
            data_time = self.data_timer.get_and_reset_avg_time()

            # add time to tensorboard
            self.log_scalar(f"{prefix}/avg_time", avg_time, it)
            self.log_scalar(f"{prefix}/data_time", data_time, it)

            est = self.batch_timer.get_est_remaining(it)
            est = datetime.timedelta(seconds=est)
            if est.days > 0:
                remaining_str = f"{est.days:2d}d {est.seconds // 3600:2d}h"
            else:
                remaining_str = (
                    f"{est.seconds // 3600:2d}h {(est.seconds % 3600) // 60:2d}m"
                )
            time_msg = f"avg_time:{avg_time:.3f} data:{data_time:.3f} remaining:{remaining_str} "
            msg = f"{msg} {time_msg}"

        msg = f"{msg} {metrics_msg}"
        self.py_log.info(msg)

    def log_audio(
        self,
        prefix: str,
        tag: str,
        waveform: torch.Tensor,
        it: int = None,
        *,
        sample_rate: int = 16000,
    ) -> Path:
        audio_dir = self.run_dir / prefix
        audio_dir.mkdir(exist_ok=True, parents=True)

        if it is None:
            name = f"{tag}.flac"
        else:
            name = f"{it:09d}_{tag}.flac"

        torchaudio.save(
            audio_dir / name,
            waveform.cpu().float(),
            sample_rate=sample_rate,
            channels_first=True,
        )
        return Path(audio_dir)

    def log_string(self, tag: str, x: str):
        self.py_log.info(f"{tag} - {x}")
        if self.tb_log is None:
            return
        self.tb_log.add_text(tag, x)

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)
