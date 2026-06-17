import hashlib
import logging
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm
import argparse

log = logging.getLogger()


class DownloadInfo(NamedTuple):
    url: str
    md5: str


LINKS: dict[str, DownloadInfo] = {
    "v1-16.pth": DownloadInfo(
        url="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth",
        md5="69f56803f59a549a1a507c93859fd4d7",
    ),
    "best_netG.pt": DownloadInfo(
        url="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt",
        md5="eeaf372a38a9c31c362120aba2dde292",
    ),
    "v1-44.pth": DownloadInfo(
        url="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth",
        md5="fab020275fa44c6589820ce025191600",
    ),
    "synchformer_state_dict.pth": DownloadInfo(
        url="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth",
        md5="5b2f5594b0730f70e41e549b7c94390c",
    ),
    "empty_string.pth": DownloadInfo(
        url="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth",
        md5="c3a76231c6acf2fa40159607d5d0011f",
    ),
    "music_speech_audioset_epoch_15_esc_89.98.pt": DownloadInfo(
        url="https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt",
        md5="d0dce80705ee4f79da5a1ff56ccde6f9",
    ),
}


def download_if_needed(path: PathLike):
    """
    Given path = ${dir}/${name}, download files of the name to ${dir}.
    """
    path = Path(path)
    base_name = path.name

    if base_name not in LINKS:
        raise ValueError(f"No link found for {base_name}")

    info = LINKS[base_name]

    path.parent.mkdir(parents=True, exist_ok=True)

    need_download = False
    if not path.exists():
        need_download = True
    else:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        # if hashlib.md5(open(path, "rb").read()).hexdigest() != info.md5:
        if hash_md5.hexdigest() != info.md5:
            log.warning(f"MD5 mismatch for {base_name}, re-downloading...")
            need_download = True
    if not need_download:
        return

    log.info(f"Downloading {base_name} from {info.url}...")
    try:
        r = requests.get(info.url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024 * 1024

        download_hash = hashlib.md5()

        with (
            tqdm(total=total_size, unit="iB", unit_scale=True, desc=base_name) as t,
            open(path, "wb") as f,
        ):
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
                download_hash.update(data)

        if total_size != 0 and t.n != total_size:
            raise RuntimeError(f"Error while downloading {base_name}: size mismatch")
        if download_hash.hexdigest() != info.md5:
            raise RuntimeError(f"Error while downloading {base_name}: MD5 mismatch. Expected {info.md5}, got {download_hash.hexdigest()}")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download {base_name}") from e


download_model_if_needed = download_if_needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-benchmark", default=None, type=Path)
    args = parser.parse_args()
    if args.av_benchmark is not None:
        print("Downloading (Fixing) all weights for av-benchmark...")
        download_if_needed(args.av_benchmark / "weights/music_speech_audioset_epoch_15_esc_89.98.pt")
        download_if_needed(args.av_benchmark / "weights/synchformer_state_dict.pth")
    else:
        print("Downloading (Fixing) all external weights...")
        download_if_needed("./ext_weights/v1-16.pth")
        download_if_needed("./ext_weights/best_netG.pt")
        download_if_needed("./ext_weights/v1-44.pth")
        download_if_needed("./ext_weights/synchformer_state_dict.pth")
        download_if_needed("./ext_weights/empty_string.pth")
    print("Complete.")