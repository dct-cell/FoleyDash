"""
trainer.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import Sequence

import torch
from av_bench.evaluate import evaluate
from av_bench.extract import extract
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from util.distribute import local_rank, is_rank0
from util.log_integrator import Integrator
from util.logger import TensorboardLogger
from util.time_estimator import PartialTimeEstimator, TimeEstimator
from util.video_joiner import VideoJoiner

from .model.flow_matching import FlowMatching
from .model.networks import get_model
from .model.sequence_config import CONFIG_16K, CONFIG_44K
from .model.util.features_utils import FeaturesUtils
from .model.util.parameter_groups import get_parameter_groups


class Runner:
    def __init__(
        self,
        cfg: DictConfig,
        log: TensorboardLogger,
        run_path: PathLike,
        for_training: bool = True,
        latent_mean: torch.Tensor = None,
        latent_std: torch.Tensor = None,
    ):
        self.device = torch.device("cuda")

        self.cfg: DictConfig = cfg

        self.exp_id: str = cfg.exp_id
        self.amp: bool = cfg.amp
        self.enable_grad_scaler: bool = cfg.enable_grad_scaler
        self.for_training: bool = for_training

        self.seq_cfg = {
            "16k": CONFIG_16K,
            "44k": CONFIG_44K,
        }[cfg.sample_rate]

        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration

        # setting up the model
        empty_string_feat = torch.load(self.cfg.ckpt.empty_string, weights_only=True)[0]

        self.network = get_model(
            cfg.variant,
            cfg.sample_rate,
            latent_mean=latent_mean,
            latent_std=latent_std,
            empty_string_feat=empty_string_feat,
        ).cuda()

        self.network = DDP(
            self.network,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        if cfg.compile:
            self.loss_velocity = torch.compile(self.loss_velocity)
            self.loss_consistency = torch.compile(self.loss_consistency)

        self.fm = FlowMatching(
            min_sigma=cfg.min_sigma,
            num_steps=cfg.NFE,
        )

        # ema profile
        if for_training and cfg.ema.enable and is_rank0:
            self.ema = PostHocEMA(
                self.network.module,
                sigma_rels=cfg.ema.sigma_rels,
                update_every=cfg.ema.update_every,
                checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                checkpoint_folder=cfg.ema.checkpoint_folder,
                step_size_correction=True,
            ).cuda()
            self.ema_start = cfg.ema.start
        else:
            self.ema = None

        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(cfg.seed + local_rank)

        # setting up feature extractors and VAEs
        self.features = FeaturesUtils(
            tod_vae_ckpt={"16k": cfg.ckpt.vae_16k, "44k": cfg.ckpt.vae_44k}[
                cfg.sample_rate
            ],
            bigvgan_vocoder_ckpt=cfg.ckpt.bigvgan_vocoder
            if cfg.sample_rate == "16k"
            else None,
            synchformer_ckpt=cfg.ckpt.synchformer,
            enable_conditions=True,
            mode=cfg.sample_rate,
            need_vae_encoder=False,
        )
        self.features = self.features.cuda().eval()

        if cfg.compile:
            # Re-enabled after fixing the `@torch.inference_mode()` decorator
            # bug in FeaturesUtils.decode (see features_utils.py:compile() note).
            # The patched compile() inside FeaturesUtils only wraps `decode` (the
            # one method that compiles cleanly under torch 2.9); vocode and the
            # CLIP/Synchformer encoders stay eager (separate inductor bugs or
            # external lib code). Net win: eval-time latent→mel decode ~10ms
            # vs ~30ms eager (3× speedup on this step); other eval ops unchanged.
            self.features.compile()

        self.time_schedule_mean = cfg.time_schedule.mean
        self.time_schedule_std = cfg.time_schedule.std
        if not isinstance(self.time_schedule_mean, Sequence):
            self.time_schedule_mean = [self.time_schedule_mean] * 3
            self.time_schedule_std = [self.time_schedule_std] * 3
        self.null_condition_probability = cfg.CFG.null_condition_probability
        self.cfg_scale = cfg.CFG.scale

        # setting up logging
        self.log = log
        self.run_path = Path(run_path)
        vgg_cfg = cfg.data.VGGSound
        if for_training:
            self.val_video_joiner = VideoJoiner(
                vgg_cfg.root,
                self.run_path / "val-sampled-videos",
                self.sample_rate,
                self.duration_sec,
            )
        else:
            self.test_video_joiner = VideoJoiner(
                vgg_cfg.root,
                self.run_path / "test-sampled-videos",
                self.sample_rate,
                self.duration_sec,
            )
        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        # setting up optimizer and loss
        if for_training:
            self.enter_train()
            parameter_groups = get_parameter_groups(
                self.network, cfg, print_log=is_rank0
            )
            self.optimizer = optim.AdamW(
                parameter_groups,
                lr=cfg.learning_rate,
                weight_decay=cfg.AdamW.weight_decay,
                betas=cfg.AdamW.betas,
                eps=1e-6 if self.amp else 1e-8,
                fused=True,
            )
            if self.enable_grad_scaler:
                self.scaler = torch.amp.GradScaler(init_scale=2048)
            self.clip_grad_norm = cfg.clip_grad_norm

            # linearly warmup learning rate
            linear_warmup_steps = cfg.linear_warmup_steps

            def warmup(currrent_step: int):
                return (currrent_step + 1) / (linear_warmup_steps + 1)

            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup
            )

            # setting up learning rate scheduler
            match cfg.lr_schedule.mode:
                case "constant":
                    next_scheduler = optim.lr_scheduler.LambdaLR(
                        self.optimizer, lr_lambda=lambda _: 1
                    )
                case "poly":
                    total_num_iter = cfg.iterations
                    next_scheduler = optim.lr_scheduler.LambdaLR(
                        self.optimizer,
                        lr_lambda=lambda x: (1 - (x / total_num_iter)) ** 0.9,
                    )
                case "step":
                    next_scheduler = optim.lr_scheduler.MultiStepLR(
                        self.optimizer, cfg.lr_schedule.steps, cfg.lr_schedule.gamma
                    )
                case _:
                    raise ValueError()

            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                [warmup_scheduler, next_scheduler],
                [linear_warmup_steps],
            )

            # Logging info
            self.log_interval = cfg.output_interval.log
            self.save_weights_interval = cfg.output_interval.save_weights
            self.save_checkpoint_interval = cfg.output_interval.save_checkpoint
            self.save_copy_iterations = cfg.save_copy_iterations
            self.num_iterations = cfg.num_iterations

            # update() is called when we log metrics, within the logger
            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_interval)
            # update() is called every iteration, in this script
            self.log.data_timer = PartialTimeEstimator(
                self.num_iterations, 1, ema_alpha=0.9
            )
        else:
            self.enter_val()

        self.stream_velocity = torch.cuda.Stream()
        self.stream_consistency = torch.cuda.Stream()

    def _sample(
        self,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
    ) -> torch.Tensor:
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        return a_mean + a_std * a_randn

    def _split(
        self,
        batch_size: int,
    ) -> tuple[int, float]:
        ratio = self.cfg.train_batch.vel_ratio
        split = int(ratio * batch_size)
        # assert 1 <= split <= batch_size - 1
        return split, split / batch_size

    def _midtime(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        it: int,
    ) -> torch.Tensor:
        """
        Sample an l satisfying s < l <= t - eps
        l is much closer to t
        """

        def parse_frac(num: float | str) -> float:
            if isinstance(num, str):
                return float(Fraction(num))
            return num

        schedule = self.cfg.time_schedule.midtime
        start_point = parse_frac(schedule.exponential.start_point)
        end_point = parse_frac(schedule.exponential.end_point)
        l = t + (s - t) * start_point * (
            (end_point / start_point) ** (it / self.num_iterations)
        )
        return torch.min(l, t - 1e-4)

    def _cfg_decay_scale(self, t: torch.Tensor):
        """
        t_decay is a number close to 1
        When t > t_decay, CFG scale is suppressed
        """
        t_decay = self.cfg.CFG_decay.t

        return torch.where(
            t <= t_decay,
            torch.ones_like(t),
            torch.zeros_like(t),
        )

    def _log_normal_sample(
        self,
        batch_size: int,
        rng: torch.Generator = None,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        t = torch.randn(batch_size, device=self.device, generator=rng) * std + mean
        return t.sigmoid()

    def _bitime_sample(
        self,
        batch_size: int,
        rng: torch.Generator = None,
        mean: tuple[float, float] = (0.0, 0.0),
        std: tuple[float, float] = (1.0, 1.0),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        0 < s <= t - eps, t <= 1 - eps
        """
        assert mean[0] <= mean[1], "s is supposed to be sampled <= t"

        s = self._log_normal_sample(batch_size, rng, mean[0], std[0])
        t = self._log_normal_sample(batch_size, rng, mean[1], std[1])

        # Distinguish two different sample modes
        if mean[0] == mean[1]:
            s, t = torch.min(s, t), torch.max(s, t)

        eps = 1e-4
        t.clip_(min=eps)
        s.clip_(max=t - eps)
        return t, s

    def loss_velocity(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        x0: torch.Tensor,  # [batch_size, seq_len, channels]
    ) -> torch.Tensor:
        batch_size = x0.size(0)

        mixin = self.cfg.mixin
        eps = self.cfg.loss.eps
        p = self.cfg.loss.p
        
        empty_conditions = self.network.module.get_empty_conditions(batch_size)

        t = self._log_normal_sample(
            batch_size,
            rng=self.rng,
            mean=self.time_schedule_mean[0],
            std=self.time_schedule_std[0],
        )
        t = t[:, None, None]
        x1, xt = self.fm.get_x1_xt(x0, t, rng=self.rng)

        min_sigma = self.cfg.min_sigma
        v_target = (1 - min_sigma) * x1 - x0

        samples = torch.rand(batch_size, device=self.device, generator=self.rng)
        mask = samples < self.null_condition_probability
        clip_f[mask] = self.network.module.empty_clip_feat
        sync_f[mask] = self.network.module.empty_sync_feat
        text_f[mask] = self.network.module.empty_string_feat

        v_pred = self.network(xt, clip_f, sync_f, text_f, t, t)

        with torch.no_grad():
            v_uncond = self.network.module.predict_flow(xt, t, t, empty_conditions)
            cfg_decay_scale = self._cfg_decay_scale(t)
            v_cond = self.network(xt, clip_f, sync_f, text_f, t, t)
            v_guided = v_target + (self.cfg_scale - 1) * cfg_decay_scale * (v_target - v_uncond)
            v_guided = mixin * v_guided + (1 - mixin) * v_cond
            v_mixin = torch.where(
                mask[:, None, None],
                v_target,
                v_guided,
            )

        delta_sqr = (v_pred - v_mixin).pow(2).flatten(1).mean(1)

        w = 1.0 / (delta_sqr + eps).pow(p)
        loss = (w.detach() * delta_sqr).mean()
        loss_mse = delta_sqr.mean()
        loss_weight = w.mean()
        return loss, loss_mse, loss_weight

    def loss_consistency(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        x0: torch.Tensor,
        it: int,
    ):
        batch_size = x0.size(0)

        mixin = self.cfg.mixin
        eps = self.cfg.loss.eps
        p = self.cfg.loss.p

        empty_conditions = self.network.module.get_empty_conditions(batch_size)

        t, s = self._bitime_sample(
            batch_size,
            rng=self.rng,
            mean=self.time_schedule_mean[1:],
            std=self.time_schedule_std[1:],
        )
        t, s = t[:, None, None], s[:, None, None]
        l = self._midtime(t, s, it)  # noqa: E741
        x1, xt = self.fm.get_x1_xt(x0, t, rng=self.rng)

        min_sigma = self.cfg.min_sigma
        v_target = (1 - min_sigma) * x1 - x0

        samples = torch.rand(batch_size, device=self.device, generator=self.rng)
        mask = samples < self.null_condition_probability

        clip_f[mask] = self.network.module.empty_clip_feat
        sync_f[mask] = self.network.module.empty_sync_feat
        text_f[mask] = self.network.module.empty_string_feat

        rng_state = self.rng.graphsafe_get_state()

        u_ts = self.network(xt, clip_f, sync_f, text_f, t, s)

        with torch.no_grad():
            v_uncond = self.network.module.predict_flow(xt, t, t, empty_conditions)
            cfg_decay_scale = self._cfg_decay_scale(t)
            v_cond = self.network(xt, clip_f, sync_f, text_f, t, t)
            v_guided = v_target + (self.cfg_scale - 1) * cfg_decay_scale * (v_target - v_uncond)

            v_guided = mixin * v_guided + (1 - mixin) * v_cond
            v_masked = torch.where(
                mask[:, None, None],
                v_target,
                v_guided,
            )

        with torch.no_grad():
            xl = xt + (l - t) * v_masked
            self.rng.graphsafe_set_state(rng_state)
            u_ls = self.network(xl, clip_f, sync_f, text_f, l, s)
            coef = (l - t) / (s - t)
            u_target = coef * v_masked + (1 - coef) * u_ls

        delta = (u_ts - u_target).flatten(1)
        w = 1.0 / (delta.pow(2).mean(1) + eps).pow(p)[:, None]
        loss = (w.detach() * delta.pow(2)).mean()
        loss_mse = delta.pow(2).mean()
        loss_weight = w.mean()
        return loss, loss_mse, loss_weight

    def train_pass(self, data, it: int = 0):
        assert self.for_training, "train_pass() should not be called when not training."

        self.enter_train()
        with torch.amp.autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
            clip_f = data["clip_features"].cuda(non_blocking=True)
            sync_f = data["sync_features"].cuda(non_blocking=True)
            text_f = data["text_features"].cuda(non_blocking=True)
            video_exist = data["video_exist"].cuda(non_blocking=True)
            text_exist = data["text_exist"].cuda(non_blocking=True)
            a_mean = data["a_mean"].cuda(non_blocking=True)
            a_std = data["a_std"].cuda(non_blocking=True)
            # these masks are for non-existent data; masking for CFG training is in loss_velocity
            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            samples = torch.rand(
                video_exist.size(0), device=self.device, generator=self.rng
            )
            mask = samples < self.null_condition_probability
            text_to_be_masked = torch.bitwise_and(mask, video_exist)
            text_f[text_to_be_masked] = self.network.module.empty_string_feat

            self.log.data_timer.end()

            x0 = self._sample(a_mean, a_std)
            x0 = self.network.module.normalize(x0)

            split, ratio = self._split(clip_f.size(0))

            if split == 0:
                # consistency only
                loss_c, loss_c_mse, weight_c = self.loss_consistency(
                    clip_f,
                    sync_f,
                    text_f,
                    x0,
                    it,
                )
                loss = loss_c
                self.train_integrator.add_dict(
                    {
                        "loss_c": loss_c,
                        "mse_c": loss_c_mse,
                        "w_c": weight_c,
                    }
                )
            elif split == clip_f.size(0):
                # velocity only
                loss_v, loss_v_mse, weight_v = self.loss_velocity(
                    clip_f,
                    sync_f,
                    text_f,
                    x0,
                )
                loss = loss_v
                self.train_integrator.add_dict(
                    {
                        "loss_v": loss_v,
                        "mse_v": loss_v_mse,
                        "w_v": weight_v,
                    }
                )
            else:
                with torch.cuda.stream(self.stream_velocity):
                    loss_v, loss_v_mse, weight_v = self.loss_velocity(
                        clip_f[:split],
                        sync_f[:split],
                        text_f[:split],
                        x0[:split],
                    )
                with torch.cuda.stream(self.stream_consistency):
                    loss_c, loss_c_mse, weight_c = self.loss_consistency(
                        clip_f[split:],
                        sync_f[split:],
                        text_f[split:],
                        x0[split:],
                        it,
                    )
                torch.cuda.current_stream().wait_stream(self.stream_velocity)
                torch.cuda.current_stream().wait_stream(self.stream_consistency)
                loss = ratio * loss_v + (1 - ratio) * loss_c
                self.train_integrator.add_dict(
                    {
                        "loss_v": loss_v,
                        "loss_c": loss_c,
                        "mse_v": loss_v_mse,
                        "mse_c": loss_c_mse,
                        "w_v": weight_v,
                        "w_c": weight_c,
                    }
                )

        if it % self.log_interval == 0 and it != 0:
            self.train_integrator.add_scalar("lr", self.scheduler.get_last_lr()[0])
            self.train_integrator.finalize("train", it)
            self.train_integrator.reset()

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad_norm
            )
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar("grad_norm", grad_norm)

        self.enter_val()

        # Save network weights and checkpoint if needed
        save_copy = it in self.save_copy_iterations

        if (it % self.save_weights_interval == 0 and it != 0) or save_copy:
            self.save_weights(it)

        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    @torch.inference_mode()
    def inference_pass(
        self, data, it: int, data_cfg: DictConfig, *, save_eval: bool = True
    ) -> Path:
        self.enter_val()
        with torch.amp.autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
            clip_f = data["clip_features"].cuda(non_blocking=True)
            sync_f = data["sync_features"].cuda(non_blocking=True)
            text_f = data["text_features"].cuda(non_blocking=True)
            video_exist = data["video_exist"].cuda(non_blocking=True)
            text_exist = data["text_exist"].cuda(non_blocking=True)
            a_mean = data["a_mean"].cuda(non_blocking=True)  # for the shape only

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            # sample
            x0 = torch.empty_like(a_mean).normal_(generator=self.rng)
            conditions = self.network.module.preprocess_conditions(
                clip_f, sync_f, text_f
            )
            cfg_ode_wrapper = lambda t, s, x: self.network.module.ode_wrapper(  # noqa: E731
                t, s, x, conditions, 0
            )
            x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
            x1_hat = self.network.module.unnormalize(x1_hat)
            mel = self.features.decode(x1_hat)
            audio = self.features.vocode(mel).cpu()
            for i in range(audio.size(0)):
                video_id = data["id"][i]
                if (not self.for_training) and i == 0:
                    # save very few videos
                    self.test_video_joiner.join(
                        video_id, f"{video_id}", audio[i].transpose(0, 1)
                    )

                # validation
                if save_eval:
                    iter_naming = f"{it:09d}"
                else:
                    iter_naming = "val-cache"
                audio_dir = self.log.log_audio(
                    iter_naming,
                    f"{video_id}",
                    audio[i],
                    it=None,
                    sample_rate=self.sample_rate,
                )
                if save_eval and i == 0 and self.for_training:
                    self.val_video_joiner.join(
                        video_id,
                        f"{iter_naming}-{video_id}",
                        audio[i].transpose(0, 1),
                    )

        return Path(audio_dir)

    @torch.inference_mode()
    def eval(
        self, 
        audio_dir: Path, 
        it: int, 
        data_cfg: DictConfig, 
        skip_clap: bool=False,
        skip_video_related: bool=False,
    ) -> dict[str, float]:
        with torch.amp.autocast("cuda", enabled=False):
            if is_rank0:
                extract(
                    audio_path=audio_dir,
                    output_path=audio_dir / "cache",
                    device="cuda",
                    batch_size=32,
                    audio_length=8,
                )
                output_metrics = evaluate(
                    gt_audio_cache=Path(data_cfg.gt_cache),
                    pred_audio_cache=audio_dir / "cache",
                    skip_clap=skip_clap,
                    skip_video_related=skip_video_related,
                )
                for k, v in output_metrics.items():
                    self.log.log_scalar(f"{data_cfg.tag}/{k}", v, it)
                    self.log.info(f"{data_cfg.tag}/{k:<10}: {v:.10f}")
            else:
                output_metrics = None

        return output_metrics

    def save_weights(self, it, save_copy=False):
        if local_rank != 0:
            return

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f"{self.exp_id}_{it}.pth"
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f"Network weights saved to {model_path}.")

        # if last exists, move it to a shadow copy
        model_path = self.run_path / f"{self.exp_id}_last.pth"
        if model_path.exists():
            shadow_path = model_path.with_name(
                model_path.name.replace("last", "shadow")
            )
            model_path.replace(shadow_path)
            self.log.info(f"Network weights shadowed to {shadow_path}.")

        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f"Network weights saved to {model_path}.")

    def save_checkpoint(self, it, save_copy=False):
        if local_rank != 0:
            return

        checkpoint = {
            "it": it,
            "weights": self.network.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f"{self.exp_id}_ckpt_{it}.pth"
            torch.save(checkpoint, model_path)
            self.log.info(f"Checkpoint saved to {model_path}.")

        # if ckpt_last exists, move it to a shadow copy
        model_path = self.run_path / f"{self.exp_id}_ckpt_last.pth"
        if model_path.exists():
            shadow_path = model_path.with_name(
                model_path.name.replace("last", "shadow")
            )
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f"Checkpoint shadowed to {shadow_path}.")

        torch.save(checkpoint, model_path)
        self.log.info(f"Checkpoint saved to {model_path}.")

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f"{self.exp_id}_ckpt_last.pth"
        if not ckpt_path.exists():
            # info_if_rank_zero(self.log, f"No checkpoint found at {ckpt_path}.")
            return None
        return ckpt_path

    def get_latest_weight_path(self):
        weight_path = self.run_path / f"{self.exp_id}_last.pth"
        if not weight_path.exists():
            self.log.info(f"No weight found at {weight_path}.")
            return None
        return weight_path

    def get_final_ema_weight_path(self):
        weight_path = self.run_path / f"{self.exp_id}_ema_final.pth"
        if not weight_path.exists():
            self.log.info(f"No weight found at {weight_path}.")
            return None
        return weight_path

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = "cuda:%d" % local_rank
        checkpoint = torch.load(
            path, map_location={"cuda:0": map_location}, weights_only=False
        )

        it = checkpoint["it"]
        weights = checkpoint["weights"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
            self.log.info(f"EMA states loaded from step {self.ema.step}")

        map_location = "cuda:%d" % local_rank
        self.network.module.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info(f"Global iteration {it} loaded.")
        self.log.info("Network weights, optimizer states, and scheduler states loaded.")

        return it

    def load_weights_in_memory(self, src_dict):
        self.network.module.load_weights(src_dict)
        self.log.info("Network weights loaded from memory.")

    def load_weights(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = "cuda:%d" % local_rank
        src_dict = torch.load(
            path, map_location={"cuda:0": map_location}, weights_only=True
        )

        self.log.info(f"Importing network weights from {path}...")
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self
