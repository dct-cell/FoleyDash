"""FoleyDash L2 pipeline verification.

Usage (from this directory, with the project conda env active):
    python verify_pipeline.py

Expected: ./out/A_dog_barking.flac (1, 64000) @ 16kHz, exit code 0.

Audio is noise / saturated -- the random model weights have no semantics
(README does not ship a trained ckpt). This script only verifies that
deps + ext_weights + CLIP + Synchformer + VAE + BigVGAN forward all
work end-to-end. ~30s on a single GPU.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANDOM_CKPT = Path("/tmp/foleydash_random_init.pth")
EXT = ROOT / "ext_weights"
OUT_DIR = ROOT / "out"
OUT_FLAC = OUT_DIR / "A_dog_barking.flac"

REQUIRED_EXT = [
    "v1-16.pth",
    "best_netG.pt",
    "synchformer_state_dict.pth",
    "empty_string.pth",
]


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


# 1. env sanity
section("[1/4] env sanity")
import torch
import torchaudio
import torchvision

print(f"  torch       {torch.__version__}  cuda? {torch.cuda.is_available()}  devices: {torch.cuda.device_count()}")
print(f"  torchaudio  {torchaudio.__version__}")
print(f"  torchvision {torchvision.__version__}")
if not torch.cuda.is_available():
    fail("CUDA not available")

# 2. ext_weights
section("[2/4] ext_weights")
missing = [f for f in REQUIRED_EXT if not (EXT / f).exists()]
if missing:
    print(f"  missing: {missing}")
    fail("run: python util/download.py")
for f in REQUIRED_EXT:
    sz = (EXT / f).stat().st_size / 1e6
    print(f"  [OK] {f}  ({sz:.1f} MB)")

# 3. random ckpt (reuse if cached)
section("[3/4] random-init ckpt")
if RANDOM_CKPT.exists():
    sz = RANDOM_CKPT.stat().st_size / 1e6
    print(f"  reuse {RANDOM_CKPT}  ({sz:.1f} MB)")
else:
    sys.path.insert(0, str(ROOT))
    from src.model.networks import get_model

    net = get_model("small", "16k")
    n = sum(p.numel() for p in net.parameters())
    torch.save(net.state_dict(), RANDOM_CKPT)
    print(f"  saved {RANDOM_CKPT}  ({n / 1e6:.1f}M params)")

# 4. run demo
section("[4/4] run demo (text-only, 4s, 4 steps)")
if OUT_FLAC.exists():
    OUT_FLAC.unlink()
cmd = [
    sys.executable,
    "demo.py",
    "--variant", "small",
    "--sample-rate", "16k",
    "--model-path", str(RANDOM_CKPT),
    "--no-download",
    "--prompt", "A dog barking",
    "--duration", "4",
    "--num-steps", "4",
    "--output", str(OUT_DIR),
]
print(f"  $ {' '.join(cmd)}")
print()
r = subprocess.run(cmd, cwd=ROOT)
if r.returncode != 0:
    fail(f"demo.py exited {r.returncode}")
if not OUT_FLAC.exists():
    fail(f"{OUT_FLAC} was not produced")

# verify audio shape
wav, sr = torchaudio.load(str(OUT_FLAC))
got_dur = wav.shape[-1] / sr
ok = sr == 16000 and abs(got_dur - 4.0) < 0.05 and wav.shape[0] == 1

section("result")
print(f"  {OUT_FLAC}")
print(f"  shape={tuple(wav.shape)}  sr={sr}  duration={got_dur:.2f}s")
if not ok:
    fail("audio shape mismatch (expected (1, 64000) @ 16kHz)")

print()
print("[PASS] all checks ok -- pipeline works end-to-end")
print("       (audio content is random, not a trained model)")
