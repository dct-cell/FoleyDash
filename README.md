# FoleyDash: Efficient Few-Step Video-to-Audio Generation

FoleyDash is a multimodal **video-to-audio (V2A)** foley generator: given a
silent video (and an optional text prompt), it synthesizes audio that is
semantically and temporally aligned with the visuals — in as few as **one
sampling step**.

🔊 **[Live demo →](https://dct-cell.github.io/FoleyDash/demo/)** — ground truth
vs. FoleyDash at 4 and 1 steps, on VGGSound test clips.

The model is a multimodal DiT trained on VAE latents with a flow-matching
objective. Instead of regressing the instantaneous velocity, it learns an
**interval-averaged velocity**, so one large step can replace many small ones —
which is what makes few-step inference work. The backbone follows
[MMAudio](https://github.com/hkchengrex/MMAudio); the contribution of this
repository is the few-step training objective and sampler.

## Installation

```bash
conda create -n foleydash python=3.12 -y
conda activate foleydash
pip install -r requirements.txt          # inference / demo
pip install -r requirements-train.txt    # extra training dependencies
```

A recent CUDA-enabled PyTorch build is required. External pretrained weights
(TOD VAE, BigVGAN vocoder, Synchformer) are **not** bundled — place them under
`ext_weights/` and point `cfg/base.yaml: EXT_WEIGHTS` there. CLIP weights are
downloaded automatically on first use.

## Usage

All entry points run through `torchrun`; data and output paths are configured in
`cfg/*.yaml`.

**1. Extract features** — done once, offline, so the training loop reads `.h5`
files and never touches raw video. Set `DATASET` / `H5` in `cfg/base.yaml`, then:

```bash
torchrun --nproc-per-node 8 latent/extract_audio.py
torchrun --nproc-per-node 8 latent/extract_video.py
```

**2. Train** — resumes automatically from the latest checkpoint in `--rundir`.
An EMA checkpoint is synthesized at the end.

```bash
torchrun --standalone --nproc_per_node=8 train.py --rundir /path/to/run
```

**3. Evaluate** — set `weight_path` and `output_name` in `cfg/eval.yaml`, then
generate audio for the whole evaluation set. Sampling steps are controlled by
`cfg/base.yaml: NFE` (e.g. `1` or `4`).

```bash
torchrun --nproc-per-node 8 eval.py
```

**4. Single-clip demo** — `--video` is optional; without it, FoleyDash runs in
text-to-audio mode.

```bash
python demo.py --duration 8 --num-steps 4 \
    --model-path /path/to/checkpoint.pth \
    --prompt "playing drums" --video /path/to/clip.mp4
```

## Acknowledgements

Built on [MMAudio](https://github.com/hkchengrex/MMAudio). The external
components (VAE, BigVGAN, Synchformer) retain their original licenses, included
alongside their code under `src/ext/`.
