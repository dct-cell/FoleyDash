import logging
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict
from torch import distributed
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from src.data.data_setup import setup_test_datasets, setup_training_datasets
from src.model.sequence_config import CONFIG_16K, CONFIG_44K
from src.runner import Runner
from src.sample import sample
from util import warn_suppress
from util.config import backup_cfg, get_rundir, load_config
from util.distribute import local_rank, world_size, is_rank0
from util.logger import TensorboardLogger
from util.synthesize_ema import synthesize_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@record
def train():
    torch.cuda.set_device(local_rank)
    from datetime import timedelta
    # Perf: pass device_id so NCCL knows which GPU each rank owns — avoids
    # "using GPU X as device used by this process is currently unknown" warning
    # and the extra device-sync overhead per collective op.
    distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=2),
        device_id=torch.device("cuda", local_rank),
    )
    if is_rank0:
        backup_cfg()
    distributed.barrier()

    run_dir = get_rundir()
    cfg = load_config("base", "data", "train")
    eval_cfg = load_config("base", "data", "train", "eval")

    # --rundir CLI overrides util/config.get_rundir() but does NOT update the
    # loaded cfg's `rundir` field, which `cfg/base.yaml` defines as
    # `./out/${exp_id}`. OmegaConf interpolations like
    # `cfg.ema.checkpoint_folder = ${rundir}/ema_ckpts` therefore resolve to
    # `./out/default/ema_ckpts` (the YAML default), not the absolute --rundir.
    # That causes PostHocEMA to try saving into a non-existent / wrong-permission
    # path at it=500 (RuntimeError "File out/default/ema_ckpts/0.500.pt cannot
    # be opened"). Patch the cfg's rundir to the resolved run_dir so the lazy
    # interpolation re-evaluates correctly everywhere downstream.
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.rundir = str(run_dir)
    with open_dict(eval_cfg):
        eval_cfg.rundir = str(run_dir)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    num_gpus = world_size

    # Perf: prefer Flash SDP but keep mem_efficient/math enabled as fallback —
    # forcing Flash-only breaks VAE attention (vae_modules.py:73) at post-train
    # sample(): RuntimeError "No available kernel" because some attn shapes are
    # not flash-friendly. With all three on, PyTorch picks Flash when it can.
    torch.backends.cuda.enable_flash_sdp(True)
    # NOTE: do NOT disable mem_efficient/math — VAE needs them as fallback
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(False)

    # patch data dim
    seq_cfg = {
        "16k": CONFIG_16K,
        "44k": CONFIG_44K,
    }[cfg.sample_rate]

    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # patch eval_cfg too — sample() / periodic eval go through ExtractedAudio
    # which reads cfg.data_dim["clip_seq_len"]; eval_cfg was loaded separately
    # and would otherwise miss these keys.
    with open_dict(eval_cfg):
        eval_cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        eval_cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        eval_cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(
        cfg.exp_id,
        run_dir,
        logging.getLogger(),
        is_rank0=is_rank0,
    )

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    cfg.batch_size //= num_gpus

    # determine time to change max skip
    total_iterations = cfg.num_iterations

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)

    eval_loader = setup_test_datasets(cfg)

    val_cfg = cfg.data[eval_cfg.dataset]
    # compute and set mean and std
    latent_mean, latent_std = dataset.compute_latent_stats()

    # construct the trainer
    trainer = Runner(
        cfg,
        log=log,
        run_path=run_dir,
        for_training=True,
        latent_mean=latent_mean,
        latent_std=latent_std,
    ).enter_train()
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # load previous checkpoint if needed
    if cfg.checkpoint is not None:
        curr_iter = trainer.load_checkpoint(cfg.checkpoint)
        cfg.checkpoint = None
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg.weights is not None:
                trainer.load_weights(cfg.weights)
                cfg.weights = None

    # determine max epoch
    current_epoch = curr_iter // len(loader)

    # training loop
    try:
        # Need this to select random bases in different workers
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            # Crucial for randomness!
            sampler.set_epoch(current_epoch)
            current_epoch += 1
            log.debug(f"Current epoch: {current_epoch}")
            trainer.enter_train()
            trainer.log.data_timer.start()
            for data in loader:
                trainer.train_pass(data, curr_iter)

                if (curr_iter + 1) % cfg.output_interval.eval == 0:
                    save_eval = (curr_iter + 1) % cfg.output_interval.save_eval == 0
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    for data in tqdm(eval_loader, position=local_rank):
                        audio_path = trainer.inference_pass(
                            data, curr_iter, val_cfg, save_eval=save_eval
                        )
                    distributed.barrier()
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)
                    trainer.eval(
                        audio_path,
                        curr_iter,
                        val_cfg,
                        skip_clap=(eval_cfg.dataset == "ExtractedVGG_test"),
                        skip_video_related=(eval_cfg.dataset == "audiocaps"),
                    )
                    distributed.barrier()  # wait for rank 0 av_bench.evaluate to finish before any rank enters next train_pass

                curr_iter += 1

                if curr_iter >= total_iterations:
                    break
    except Exception as e:
        log.error(f"Error occurred at iteration {curr_iter}!")
        log.critical(e.message if hasattr(e, "message") else str(e))
        raise
    finally:
        trainer.save_checkpoint(curr_iter)
        trainer.save_weights(curr_iter)

    # Inference pass
    del trainer
    torch.cuda.empty_cache()

    # Synthesize EMA. On any failure other than "EMA ckpt folder is empty"
    # (which is a legitimate state for short smoke runs where
    # cfg.ema.checkpoint_every > num_iterations), abort — sample() falls back to
    # default_last.pth when ema_final is missing, but if synthesize_ema raises
    # mid-pipeline (e.g. corrupt snapshot, strict-load mismatch from torch.compile
    # prefix surgery) we want to surface it loudly rather than ship random-quality
    # samples downstream.
    if is_rank0:
        ema_sigma = cfg.ema.default_output_sigma
        log.info(f"Synthesizing EMA with sigma={ema_sigma}")
        save_dir = Path(run_dir) / f"{cfg.exp_id}_ema_final.pth"
        try:
            state_dict = synthesize_ema(cfg, ema_sigma, step=None)
            torch.save(state_dict, save_dir)
            log.info(f"Synthesized EMA saved to {save_dir}!")
            assert save_dir.exists(), f"torch.save claimed success but {save_dir} missing"
        except ValueError as e:
            # nitrous_ema.synthesize_ema_model raises ValueError("max() ... empty")
            # when no EMA snapshots exist — expected for very short smoke runs.
            if "empty" in str(e):
                log.warning(
                    f"No EMA snapshots in {cfg.ema.checkpoint_folder} "
                    f"(num_iterations < ema.checkpoint_every?); skipping ema_final.pth"
                )
            else:
                raise
    distributed.barrier()

    # Bug 8 fix: sample.py uses `cfg.rundir` internally (loaded from base.yaml as
    # `./out/${exp_id}`), which does NOT pick up the --rundir CLI override that
    # only patches util/config.get_rundir(). Explicitly inject get_rundir() here
    # so that sample()/Runner find the actual EMA weights and write artifacts
    # to the right rundir.
    with open_dict(eval_cfg):
        eval_cfg.rundir = str(run_dir)
    sample(eval_cfg)

    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    train()
