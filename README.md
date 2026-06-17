# FoleyDash: Efficient Few-Step Video-to-Audio Generation

FoleyDash is a multimodal **video-to-audio (V2A)** foley generator. Given a
silent video and an optional text prompt, it synthesizes audio that is
semantically and temporally aligned with the visual content, using only a
handful of sampling steps (and remaining usable even at a single step).

The model is a multimodal DiT trained on VAE latents with a flow-matching
objective. Instead of regressing the instantaneous velocity, FoleyDash learns
an **interval-averaged velocity** so that one large step can replace many small
ones. The training target is built entirely from forward passes (no
Jacobian-vector products), classifier-free guidance is folded into the training
loss so inference needs only one pass per step, and the time conditioning is
extended to encode the size of each jump.

The backbone follows [MMAudio](https://github.com/hkchengrex/MMAudio); the
contribution of this repository is the training objective and the few-step
sampler.

## Repository layout

```
src/            model, runner, flow matching, multimodal backbone, feature utils
  model/        the FoleyDash network and flow-matching sampler
  data/         training and evaluation datasets
  ext/          external encoders/decoders (VAE, BigVGAN, Synchformer) — code only
latent/         offline feature extraction (CLIP / Synchformer / VAE -> .h5)
cfg/            base / data / train / eval configs (YAML, merged by util/config.py)
util/           config, distributed, logging, EMA synthesis helpers
train.py        training entry point
eval.py         batch generation on an evaluation set
demo.py         single-clip inference (text + optional video -> audio)
```

## Installation

```bash
conda create -n foleydash python=3.12 -y
conda activate foleydash
pip install -r requirements.txt          # inference / demo
pip install -r requirements-train.txt    # additional training deps
```

A recent PyTorch build with CUDA is required. External pretrained weights (the
TOD VAE, the BigVGAN vocoder, the Synchformer encoder, and the empty-string
embedding) are **not** included in this repository; place them under
`ext_weights/` and point `cfg/base.yaml: EXT_WEIGHTS` at that directory. CLIP
weights are downloaded automatically on first use.

## Data preparation

FoleyDash trains on VAE/CLIP/Synchformer features that are extracted **once**,
offline, into `.h5` files, so the training loop never touches raw video. Set the
dataset and output paths in `cfg/base.yaml` (`DATASET`, `H5`) and the per-dataset
entries in `cfg/data.yaml`, then run:

```bash
torchrun --nproc-per-node 8 latent/extract_audio.py
torchrun --nproc-per-node 8 latent/extract_video.py
```

The TSV file lists that index each dataset are not shipped here; see
`cfg/data.yaml` for the expected columns. We follow the data setup of MMAudio
(VGGSound for video-text-audio, plus AudioCaps / Clotho / WavCaps for
text-audio).

## Training

```bash
torchrun --standalone --nproc_per_node=8 train.py --rundir /path/to/run
```

Training resumes automatically from the latest checkpoint in `--rundir`.
Evaluation runs periodically during training (every `cfg.output_interval.eval`
steps) and computes metrics on the held-out set. Key settings live in
`cfg/train.yaml` (steps, batch size, learning-rate schedule) and `cfg/base.yaml`
(guidance scale, anchor ratio, blend schedule, etc.). After training, an
exponential-moving-average checkpoint is synthesized automatically.

## Evaluation

Set `weight_path` and `output_name` in `cfg/eval.yaml`, then generate audio for
the whole evaluation set:

```bash
torchrun --nproc-per-node 8 eval.py
```

The number of sampling steps is controlled by `cfg/base.yaml: NFE` (e.g. `1` or
`4`).

## Single-clip demo

```bash
python demo.py --duration 8 --num-steps 4 \
    --model-path /path/to/checkpoint.pth \
    --prompt "playing drums" --video /path/to/clip.mp4
```

`--video` is optional; without it, the model runs in text-to-audio mode.

## Acknowledgements

The multimodal backbone builds on [MMAudio](https://github.com/hkchengrex/MMAudio),
and the external components (VAE, BigVGAN, Synchformer) retain their original
licenses, included alongside their code under `src/ext/`.
