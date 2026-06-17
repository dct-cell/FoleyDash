from os import PathLike
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from util.distribute import local_rank, is_rank0


def create_h5(
    path: PathLike,
    mode: Literal["video", "audio"],
    seq_mode: Literal["16k", "44k"],
    expected_size: int,
    expected_batch_size: int = None,
):
    match seq_mode:
        case "16k":
            latent_dim = 20
            clip_dim = 1024
            sync_dim = 768
            text_dim = 1024
            latent_seq_len = 250
            clip_seq_len = 64
            sync_seq_len = 192
        case "44k":
            latent_dim = 40
            clip_dim = 1024
            sync_dim = 768
            text_dim = 1024
            latent_seq_len = 345
            clip_seq_len = 64
            sync_seq_len = 192
        case _:
            raise RuntimeError(f"seq_mode {seq_mode} not supported")

    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        # File name, without ext
        f.create_dataset(
            "id",
            expected_size,
            dtype=dt,
            chunks=True,
        )
        # Text label in string
        f.create_dataset(
            "label",
            expected_size,
            dtype=dt,
            chunks=True,
        )

        f.create_dataset(
            "mean",
            (expected_size, latent_seq_len, latent_dim),
            dtype=np.float32,
            chunks=(expected_batch_size, latent_seq_len, latent_dim) or True,
        )
        f.create_dataset(
            "std",
            (expected_size, latent_seq_len, latent_dim),
            dtype=np.float32,
            chunks=(expected_batch_size, latent_seq_len, latent_dim) or True,
        )
        if mode == "video":
            f.create_dataset(
                "clip_features",
                (expected_size, clip_seq_len, clip_dim),
                dtype=np.float32,
                chunks=(expected_batch_size, clip_seq_len, clip_dim) or True,
            )
            f.create_dataset(
                "sync_features",
                (expected_size, sync_seq_len, sync_dim),
                dtype=np.float32,
                chunks=(expected_batch_size, sync_seq_len, sync_dim) or True,
            )
        f.create_dataset(
            "text_features",
            (expected_size, 77, text_dim),
            dtype=np.float32,
            chunks=(expected_batch_size, 77, text_dim) or True,
        )


def merge_h5_copy(path: PathLike, src: Sequence[PathLike], *, remove_blob: bool=False):
    assert len(src) >= 1

    sizes = []
    for path_src in src:
        with h5py.File(path_src, "r") as f:
            sizes.append(f["id"].shape[0])
    
    total_len = sum(sizes)
    print(f"Total samples to merge: {total_len}")

    with h5py.File(path, "w") as merged:
        with h5py.File(src[0], "r") as f_first:
            keys = list(f_first.keys())
            
            for key in keys:
                dset_src = f_first[key]
                new_shape = (total_len, *dset_src.shape[1:])
                merged.create_dataset(
                    key, 
                    shape=new_shape, 
                    dtype=dset_src.dtype
                )

        current_idx = 0
        for i, path_src in enumerate(src):
            n_samples = sizes[i]
            end_idx = current_idx + n_samples
            with h5py.File(path_src, "r") as f_src:
                for key in keys:
                    merged[key][current_idx:end_idx] = f_src[key][:]
            current_idx = end_idx
    print(f"Saved merged file to {path}")

    if remove_blob:
        print("remove_blob=True: Deleting source files...")
        for path_src in src:
            try:
                file_path = Path(path_src)
                if file_path.exists():
                    file_path.unlink()
            except OSError as e:
                print(f"Warning: Failed to remove {path_src}. Error: {e}")
        print("Source files cleanup complete.")


def merge_h5_vds(path: PathLike, src: Sequence[PathLike]):
    assert len(src) >= 1
    sizes = [0]
    for path_src in src:
        with h5py.File(path_src) as f:
            sizes.append(f["id"].shape[0])
    cumsum = np.cumsum(sizes)

    with h5py.File(path, "w") as merged, h5py.File(src[0]) as f:
        for key in f:
            dset = f[key]
            layout = h5py.VirtualLayout(
                (cumsum[-1], *dset.shape[1:]),
                dtype=dset.dtype,
            )
            for i in range(len(src)):
                begin, end = cumsum[i], cumsum[i + 1]
                layout[begin:end, ...] = h5py.VirtualSource(
                    src[i],
                    key,
                    (end - begin, *dset.shape[1:]),
                    dtype=dset.dtype,
                )
            merged.create_virtual_dataset(key, layout)


def shrink_h5(file: h5py.File, size: int):
    for name, dset in file.items():
        if not isinstance(dset, h5py.Dataset):
            continue
        old_size = dset.shape[0]
        if old_size <= size:
            continue
        dset.resize(size, axis=0)
        print(f"{name}: {old_size} -> {size}")


@torch.inference_mode()
def _test():
    Path("blob").mkdir(exist_ok=True)
    sub = Path(f"blob/{local_rank}.h5")
    create_h5(sub, "video", 100, 4)
    with h5py.File(sub, "a") as f:
        for i in tqdm(range(0, 100, 4), position=local_rank):
            for j in range(i, i + 4):
                f["id"][j] = f"id-r{local_rank}-i{j}"
            A = np.fromfunction(lambda i, j, k: i + j + k, (4, 250, 20), dtype=float)
            f["mean"][i : i + 4] = A

    dist.barrier()
    if is_rank0:
        merge_h5_vds("merged.h5", [f"blob/{local_rank}.h5" for local_rank in range(4)])
    dist.barrier()
    if is_rank0:
        with h5py.File("merged.h5") as f:
            print(f["id"])
            print(f["id"][0].decode("utf-8"))
            print(f["mean"][0:8])
    dist.barrier()


if __name__ == "__main__":
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    _test()
    dist.destroy_process_group()
