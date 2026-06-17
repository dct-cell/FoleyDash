# isort: off
import site
from pathlib import Path
site.addsitedir(str(Path(__file__).parent.parent))
from util import warn_suppress  # noqa: F401
# isort: on

import h5py
import torch
import torch.distributed as dist
import torch.nn.functional as F
from src.ext.autoencoder import AutoEncoderModule
from src.ext.mel_converter import get_mel_converter
from open_clip import create_model_from_pretrained

from src.model.sequence_config import CONFIG
from util.config import load_config
from util.distribute import local_rank, world_size, is_rank0
from util.distributed_logger import distributed_tqdm
from util.tool import sliced, to_numpy

from h5 import create_h5, merge_h5_vds, shrink_h5, merge_h5_copy
from loader import audio_loader
from omegaconf import OmegaConf


@torch.inference_mode()
def extract_audio():
    cfg = load_config("base", "data", load_root=True)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("./latent/cfg_latent.yaml"))

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    sample_rate = cfg.sample_rate

    SEQ_CFG = CONFIG[sample_rate]

    clip_model = create_model_from_pretrained(
        "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
    )
    clip_model = clip_model.eval().cuda()

    # a hack to make it output last hidden states
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)

    tod = AutoEncoderModule(
        vae_ckpt_path={
            "16k": cfg.ckpt.vae_16k,
            "44k": cfg.ckpt.vae_44k,
        }[sample_rate],
        vocoder_ckpt_path=cfg.ckpt.bigvgan_vocoder if sample_rate == "16k" else None,
        mode=sample_rate,
    )
    tod = tod.eval().cuda()
    mel_converter = get_mel_converter(sample_rate).eval().cuda()

    for name in cfg.audio_datasets:
        cfg_dataset = cfg.data[name]

        data = cfg_dataset.data
        tsv = cfg_dataset.tsv
        tsv_clip = cfg_dataset.tsv_clip
        latent_path = Path(cfg_dataset.latent)
        loader = audio_loader(
            cfg,
            data,
            tsv,
            tsv_clip,
            SEQ_CFG.sampling_rate,
            SEQ_CFG.num_audio_frames,
        )
        expected_batch_size = 64

        path_file = latent_path.parent / "blob" /  f"{latent_path.stem}-r{local_rank}.h5"
        path_file.parent.mkdir(parents=True, exist_ok=True)

        create_h5(
            path_file,
            mode="audio",
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
                f["id"][split] = data["id"]
                f["label"][split] = data["caption"]

                waveforms = data["waveform"].cuda()
                tokens = data["tokens"].cuda()

                text_features = clip_model.encode_text(tokens, normalize=True)
                mel = mel_converter(waveforms)
                encode = tod.encode(mel)

                f["mean"][split] = to_numpy(encode.mean).swapaxes(1, 2)
                f["std"][split] = to_numpy(encode.std).swapaxes(1, 2)

                f["text_features"][split] = to_numpy(text_features)
            shrink_h5(f, total)
        dist.barrier()

        if is_rank0:
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
    extract_audio()
    dist.destroy_process_group()
