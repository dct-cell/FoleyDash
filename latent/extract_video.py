# isort: off
import site
from pathlib import Path
site.addsitedir(str(Path(__file__).parent.parent))
from util import warn_suppress  # noqa: F401
# isort: on

import h5py
import torch
import torch.distributed as dist
from src.model.util.features_utils import FeaturesUtils
from h5 import create_h5, merge_h5_vds, shrink_h5, merge_h5_copy
from loader import video_loader
from src.model.sequence_config import CONFIG

from util.distribute import local_rank, world_size
from util.distributed_logger import distributed_tqdm
from util.tool import sliced, to_numpy
from util.config import load_config

from omegaconf import OmegaConf

@torch.inference_mode()
def extract_video():
    cfg = load_config("base", "data", load_root=True)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("./latent/cfg_latent.yaml"))

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    sample_rate = cfg.sample_rate

    SEQ_CFG = CONFIG[sample_rate]

    feature_extractor = FeaturesUtils(
        tod_vae_ckpt={
            "16k": cfg.ckpt.vae_16k,
            "44k": cfg.ckpt.vae_44k,
        }[sample_rate],
        enable_conditions=True,
        bigvgan_vocoder_ckpt=cfg.ckpt.bigvgan_vocoder if sample_rate == "16k" else None,
        synchformer_ckpt=cfg.ckpt.synchformer,
        mode=sample_rate,
    )
    feature_extractor = feature_extractor.eval().cuda()

    for name in cfg.video_datasets:
        cfg_dataset = cfg.data[name]

        data = cfg_dataset.data
        tsv = cfg_dataset.tsv
        latent_path = Path(cfg_dataset.latent)
        loader = video_loader(
            cfg, data, tsv, SEQ_CFG.sampling_rate, 8.0, SEQ_CFG.num_audio_frames
        )
        expected_batch_size = 64

        path_file = latent_path.parent / "blob" /  f"{latent_path.stem}-r{local_rank}.h5"
        path_file.parent.mkdir(parents=True, exist_ok=True)

        create_h5(
            path_file,
            mode="video",
            seq_mode=sample_rate,
            expected_size=len(loader) * cfg.batch_size,
            expected_batch_size=expected_batch_size,
        )

        with h5py.File(path_file, "a") as f:
            total = 0
            for split, data in sliced(
                distributed_tqdm(
                    loader,
                    desc=f"[r{local_rank}]",
                )
            ):
                if data is None:
                    continue
                total += len(data["id"])
                # log.info(f"{total}")
                f["id"][split] = data["id"]
                f["label"][split] = data["caption"]

                audio = data["audio"].cuda()
                encode = feature_extractor.encode_audio(audio)

                f["mean"][split] = to_numpy(encode.mean).swapaxes(1, 2)
                f["std"][split] = to_numpy(encode.std).swapaxes(1, 2)

                clip_video = data["clip_video"].cuda()
                clip_features = feature_extractor.encode_video_with_clip(clip_video)
                f["clip_features"][split] = to_numpy(clip_features)

                sync_video = data["sync_video"].cuda()
                sync_features = feature_extractor.encode_video_with_sync(sync_video)
                f["sync_features"][split] = to_numpy(sync_features)

                caption = data["caption"]
                text_features = feature_extractor.encode_text(caption)
                f["text_features"][split] = to_numpy(text_features)
            shrink_h5(f, total)
        dist.barrier()

        if local_rank == 0:
            merge_h5_copy(
                latent_path,
                [latent_path.parent / "blob" / f"{latent_path.stem}-r{r}.h5" for r in range(world_size)],
                remove_blob=True,
            )
            # merge_h5_vds(
            #     f"{path_h5}/{name}.h5",
            #     [f"{path_blob}/{name}-r{r}.h5" for r in range(world_size)],
            # )
        dist.barrier()


if __name__ == "__main__":
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    extract_video()
    dist.destroy_process_group()
