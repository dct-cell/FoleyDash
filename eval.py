from util import warn_suppress

import logging
import warnings
from pathlib import Path

import torch
import torchaudio
from torch import distributed
from tqdm import tqdm

from src.data.data_setup import setup_eval_dataset
from src.eval_utils import ModelConfig, all_model_cfg, generate
from src.model.flow_matching import FlowMatching
from src.model.networks import FoleyDash, get_model
from src.model.util.features_utils import FeaturesUtils
from util.config import load_config
from util.distribute import local_rank

log = logging.getLogger()


@torch.inference_mode()
def main():
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")

    cfg = load_config("data", "base", "eval", load_root=True)
    device = "cuda"

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    model: ModelConfig = all_model_cfg[f"{cfg.variant}_{cfg.sample_rate}"]
    # model.download_if_needed()
    seq_cfg = model.seq_cfg

    model_path = Path(cfg.weight_path)
    run_dir = model_path.parent

    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f"{cfg.dataset}-{cfg.output_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    seq_cfg.duration = cfg.duration_s
    net: FoleyDash = get_model(cfg.variant, cfg.sample_rate).to(device).eval()

    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    log.info(f"Loaded weights from {model_path}")
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
    )
    log.info(f"Latent seq len: {seq_cfg.latent_seq_len}")
    log.info(f"Clip seq len: {seq_cfg.clip_seq_len}")
    log.info(f"Sync seq len: {seq_cfg.sync_seq_len}")

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(
        min_sigma=cfg.min_sigma,
        num_steps=cfg.NFE,
    )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device).eval()

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    _, loader = setup_eval_dataset(cfg.dataset, cfg)

    with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
        try:
            for batch in tqdm(loader, position=local_rank):
                audios = generate(
                    batch.get("clip_video", None),
                    batch.get("sync_video", None),
                    batch.get("caption", None),
                    feature_utils=feature_utils,
                    net=net,
                    fm=fm,
                    rng=rng,
                    clip_batch_size_multiplier=64,
                    sync_batch_size_multiplier=64,
                )
                audios = audios.float().cpu()
                names = batch["name"]
                for audio, name in zip(audios, names):
                    torchaudio.save(
                        output_dir / f"{name}.flac", audio, seq_cfg.sampling_rate
                    )
        except Exception as e:
            print(f"[{local_rank}] Error: {e}")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
