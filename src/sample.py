import json
import logging
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from .data.data_setup import setup_test_datasets
from .runner import Runner

# from util.distribute import info_if_rank_zero
from util.logger import TensorboardLogger
from util.distribute import local_rank, is_rank0

# Debug: cap loader iterations when verifying sample pipeline (default 0 = no cap).
# Set FOLEYDASH_DEBUG_SAMPLE_MAX_BATCHES=N to break after N batches.
_DEBUG_SAMPLE_MAX_BATCHES = int(os.environ.get("FOLEYDASH_DEBUG_SAMPLE_MAX_BATCHES", "0"))


def sample(cfg: DictConfig):
    run_dir = cfg.rundir

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(
        cfg.exp_id,
        run_dir,
        logging.getLogger(),
        is_rank0=is_rank0,
    )

    # cuda setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # construct the trainer
    runner = Runner(cfg, log=log, run_path=run_dir, for_training=False).enter_val()

    # load the last weights if needed
    if cfg["weights"] is not None:
        runner.load_weights(cfg["weights"])
        cfg["weights"] = None
    else:
        weights = runner.get_final_ema_weight_path()
        if weights is None:
            # ema_final.pth missing — fall back to training-final weights rather
            # than letting Runner's random init silently produce garbage samples.
            # (Real fix: investigate why synthesize_ema didn't write ema_final.)
            weights = runner.get_latest_weight_path()
            assert weights is not None, (
                f"Both {cfg.exp_id}_ema_final.pth and {cfg.exp_id}_last.pth "
                f"missing from {run_dir}; sample() refuses to run with random "
                f"weights. Check train.py post-train synthesize_ema log."
            )
            log.warning(
                f"ema_final.pth missing; sampling with {weights.name} "
                f"(training-final, NOT EMA — expect 5-15% worse metrics)."
            )
        runner.load_weights(weights)

    # setup datasets
    loader = setup_test_datasets(cfg)
    data_cfg = cfg.data.ExtractedVGG_test
    with open_dict(data_cfg):
        if cfg.output_name is not None:
            # append to the tag
            data_cfg.tag = f"{data_cfg.tag}-{cfg.output_name}"

    # loop
    audio_path = None
    for curr_iter, data in enumerate(tqdm(loader)):
        if _DEBUG_SAMPLE_MAX_BATCHES > 0 and curr_iter >= _DEBUG_SAMPLE_MAX_BATCHES:
            break
        # save_eval=False makes runner.inference_pass return Path("val-cache") for every batch,
        # so the assert below holds (otherwise iter_naming=f"{curr_iter:09d}" creates a new dir each call)
        new_audio_path = runner.inference_pass(data, curr_iter, data_cfg, save_eval=False)
        if audio_path is None:
            audio_path = new_audio_path
        else:
            assert audio_path == new_audio_path, "Different audio path detected"

    # info_if_rank_zero(log, f"Inference completed. Audio path: {audio_path}")
    output_metrics = runner.eval(audio_path, curr_iter, data_cfg)

    if is_rank0:
        # write the output metrics to run_dir
        output_metrics_path = os.path.join(
            run_dir, f"{data_cfg.tag}-output_metrics.json"
        )
        with open(output_metrics_path, "w") as f:
            json.dump(output_metrics, f, indent=4)
