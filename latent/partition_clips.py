import logging
from pathlib import Path

import pandas as pd
import soundfile as sf
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from util.config import load_config


def main():
    cfg = load_config("base")
    min_length_sec = 8.1
    max_segments_per_clip = 5

    cfg_audio: DictConfig = cfg.audio
    OmegaConf.set_readonly(cfg_audio, False)
    OmegaConf.set_struct(cfg_audio, False)
    clip_tsv = Path(cfg_audio.pop("clip_tsv"))
    cfg_audio.pop("batch_size")
    cfg_audio.pop("num_workers")
    cfg_audio.pop("mode")
    cfg_audio.pop("tf32")
    OmegaConf.set_struct(cfg_audio, True)

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    for split, cfg_split in cfg_audio.items():
        data_dir = Path(cfg_split.data)
        csv_dir = Path(cfg_split.tsv)
        output_data = []
        audio_names = sorted(pd.read_csv(csv_dir, sep="\t")["id"].tolist())
        log.info(f"audio_names: Processing {len(audio_names)} files")

        for f in data_dir.iterdir():
            if f.is_file():
                try:
                    sf.info(f)
                except RuntimeError:
                    continue
                ext = f.suffix
                break

        for audio_name in tqdm(audio_names):
            audio_file_path = data_dir / f"{audio_name}{ext}"
            audio_name = audio_file_path.stem

            try:  # Seems containing some files causing errors
                info = sf.info(audio_file_path)
                sample_rate = info.samplerate
                total_length = info.frames
            except RuntimeError as e:
                log.warning(f"Failed to read {audio_name}: {e}")
                continue

            if total_length < sample_rate * min_length_sec:
                continue

            # try to partition the audio into segments, each with length of min_length_sec
            segment_length = int(sample_rate * min_length_sec)
            num_segments = min(max_segments_per_clip, total_length // segment_length)
            if num_segments > 1:
                segment_interval = (total_length - segment_length) // (num_segments - 1)
            else:
                segment_interval = 0

            for i in range(num_segments):
                start_sample = i * segment_interval
                end_sample = start_sample + segment_length
                audio_id = f"{audio_name}_{i}"
                output_data.append((audio_id, audio_name, start_sample, end_sample))

        clip_tsv.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(
            output_data, columns=["id", "name", "start_sample", "end_sample"]
        )
        output_df.to_csv(clip_tsv / f"{split}.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
