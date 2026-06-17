import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from src.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
    setup_eval_logging,
)
from src.model.flow_matching import FlowMatching
from src.model.networks import FoleyDash, get_model
from src.model.util.features_utils import FeaturesUtils

log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default="small", choices=["small"]
    )
    parser.add_argument(
        "--sample-rate", type=str, default="16k", choices=["16k"]
    )
    parser.add_argument("--video", type=Path, help="Path to the video file")
    parser.add_argument("--prompt", type=str, help="Input prompt", default="")
    parser.add_argument(
        "--negative-prompt", type=str, help="Negative prompt", default=""
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--mask-away-clip", action="store_true")
    parser.add_argument("--output", type=Path, help="Output directory", default="./out")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--skip-video-composite", action="store_true")
    parser.add_argument("--fp", action="store_true", help="Enable fp32")
    parser.add_argument("--no-tf32", action="store_true", help="Disable tf32")
    parser.add_argument(
        "--no-download", action="store_true", help="Disable downloading models online"
    )
    parser.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    dtype = torch.float32 if args.fp else torch.bfloat16

    model: ModelConfig = all_model_cfg[f"{args.variant}_{args.sample_rate}"]
    seq_cfg = model.seq_cfg
    if not args.no_download:
        model.download_if_needed()

    video_path: Path | None = args.video.expanduser() if args.video else None

    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: Path = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip
    model_path = (
        Path(args.model_path) if args.model_path is not None else model.model_path
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        log.warning("CUDA/MPS are not available, running on CPU")

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: FoleyDash = get_model(args.variant, args.sample_rate).to(device, dtype).eval()
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    log.info(f"Loaded weights from {model_path}")

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, num_steps=num_steps)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    if video_path is not None:
        log.info(f"Using video {video_path}")
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
    else:
        log.info("No video provided -- text-to-audio mode")
        clip_frames = sync_frames = None

    seq_cfg.duration = duration
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
    )

    log.info(f"Prompt: {prompt}")
    log.info(f"Negative prompt: {negative_prompt}")

    audios = generate(
        clip_frames,
        sync_frames,
        [prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
    )
    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f"{video_path.stem}.flac"
    else:
        safe_filename = prompt.replace(" ", "_").replace("/", "_").replace(".", "")
        save_path = output_dir / f"{safe_filename}.flac"
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f"Audio saved to {save_path}")
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f"{video_path.stem}.mp4"
        make_video(
            video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate
        )
        log.info(f"Video saved to {video_save_path}")

    log.info(f"Memory usage: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB")


if __name__ == "__main__":
    main()
