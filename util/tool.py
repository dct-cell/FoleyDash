import sys
from pathlib import Path
from typing import Literal

import torch

import __main__

from os import PathLike

from functools import wraps

from util.distribute import local_rank


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def sliced(loader):
    """Yield (split, batch) from a loader, with continuous slicing."""
    start = 0
    for batch in loader:
        # print(batch.shape)
        end = start + len(batch["id"])
        yield slice(start, end), batch
        start = end


def get_cfg_entry(entries, key, value):
    return next((e for e in entries if e[key] == value), None)


def path_norm(
    path: PathLike,
    *,
    ext: str = None,
    mode: Literal["cwd", "main", "find"] = "cwd",
) -> Path:
    """
    Normalize a path and return an absolute Path object.

    Parameters:
        path: str or Path
        ext: Optional file extension to append if the path has no suffix
        mode:
            cwd: relative paths are resolved from the current working directory
            main: relative paths are resolved from the entry script directory
            find: try current working directory first, then entry script directory if not found

    Returns:
        Absolute Path object
    """
    # Expand user home (~)
    path = Path(path).expanduser()

    # Append extension if specified and path has no suffix, skip if directory
    if ext is not None and not path.suffix and not path.is_dir():
        if not ext.startswith("."):
            ext = "." + ext
        path = path.with_suffix(ext)

    # If path is already absolute, just resolve and return
    if path.is_absolute():
        return path.resolve(strict=False)

    # Handle relative paths based on mode
    match mode:
        case "cwd":
            return path.resolve(strict=False)
        case "main":
            main_dir = Path(getattr(__main__, "__file__", sys.argv[0])).resolve().parent
            return (main_dir / path).resolve(strict=False)
        case "find":
            cand1 = path.resolve(strict=False)
            if cand1.exists():
                return cand1
            else:
                main_dir = (
                    Path(getattr(__main__, "__file__", sys.argv[0])).resolve().parent
                )
                cand2 = (main_dir / path).resolve(strict=False)
                return cand2 if cand2.exists() else cand1
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

def rank0_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if local_rank != 0:
            return
        return func(*args, **kwargs)
    return wrapper