from os import PathLike
from pathlib import Path
import subprocess
import tempfile

import torch
import torchaudio


class VideoJoiner:
    def __init__(
        self,
        src_root: PathLike,
        output_root: PathLike,
        sample_rate: int,
        duration_seconds: float,
    ):
        self.src_root = Path(src_root)
        self.output_root = Path(output_root)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

        self.output_root.mkdir(parents=True, exist_ok=True)

    def join(self, video_id: str, output_name: str, audio: torch.Tensor):
        video_path = self.src_root / f"{video_id}.mp4"
        output_path = self.output_root / f"{output_name}.mp4"
        merge_audio_into_video(
            video_path, output_path, audio, self.sample_rate, self.duration_seconds
        )


def merge_audio_into_video(
    video_path: PathLike,
    output_path: PathLike,
    audio: torch.Tensor,
    sample_rate: int,
    duration_seconds: float,
):
    # audio: (num_samples, num_channels=1/2)
    # Replaces torio StreamingMediaDecoder/Encoder with ffmpeg subprocess
    # (torio was removed in torchaudio 2.9; ffmpeg binary is in both conda env
    # and Docker image; this is simpler than PyAV encoder API for one-shot mux).

    audio = audio.float().cpu().contiguous()
    if audio.dim() == 1:
        audio = audio.unsqueeze(-1)
    waveform = audio.transpose(0, 1)  # (C, N) for torchaudio.save

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_audio = tmp.name
    try:
        torchaudio.save(tmp_audio, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-i", tmp_audio,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "libmp3lame",
                "-t", str(duration_seconds),
                "-shortest",
                str(output_path),
            ],
            check=True,
        )
    finally:
        Path(tmp_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    # Usage example
    import sys

    audio = torch.randn(16000 * 4, 1)
    merge_audio_into_video(sys.argv[1], sys.argv[2], audio, 16000, 4)
