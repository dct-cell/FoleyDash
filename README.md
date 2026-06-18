# FoleyDash: Efficient Few-Step Video-to-Audio Generation

FoleyDash is a multimodal **video-to-audio (V2A)** foley generator: given a
silent video (and an optional text prompt), it synthesizes audio that is
semantically and temporally aligned with the visuals, in as few as **one
sampling step**.

🔊 **[Live demo →](https://dct-cell.github.io/FoleyDash/demo/)**: ground truth
vs. FoleyDash at 4 and 1 steps, on VGGSound test clips.

The model is a multimodal DiT trained on VAE latents with a flow-matching
objective. Instead of regressing the instantaneous velocity, it learns an
**interval-averaged velocity**, so one large step can replace many small ones,
which is what makes few-step inference work. The backbone follows
[MMAudio](https://github.com/hkchengrex/MMAudio); the contribution of this
repository is the few-step training objective and sampler.

## Installation

```bash
conda create -n foleydash python=3.12 -y
conda activate foleydash
# Install a CUDA build of PyTorch first (tested with torch 2.9 + cu130)
pip install torch torchaudio torchvision
pip install -r requirements.txt
```

Then install **ffmpeg 7** so that `torchcodec` can decode and mux video
(it links against `libavutil.so.59`, which ships with ffmpeg 7, not the
ffmpeg 4 in most apt repos):

```bash
conda install -c conda-forge "ffmpeg>=7" -y
```

The last four entries in `requirements.txt` (installed from Git) are the
backbone networks used to compute evaluation metrics; training and the demo
run without them.

### External weights

FoleyDash builds on a few frozen pretrained modules (TOD VAE, BigVGAN vocoder,
Synchformer, and an empty-string text embedding). These are **not** bundled in
the repository. Download them into `ext_weights/` with:

```bash
python util/download.py
```

CLIP weights are fetched automatically from Hugging Face on first use. Keep
`cfg/base.yaml: EXT_WEIGHTS` pointing at this directory (default `./ext_weights`).

To check that the whole forward pipeline (dependencies, external weights, CLIP,
Synchformer, VAE, vocoder) works end to end without a trained checkpoint, run
`python verify_pipeline.py`. It generates a short clip from a random-init model
and verifies the output shape; the audio content is noise by design.

## Data preparation

You only need this section to **train** or to run **batch evaluation**. The
single-clip demo encodes its input on the fly and needs no dataset.

### 1. Download datasets

FoleyDash is trained on a mix of video and audio-text datasets:
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) (video),
[AudioCaps](https://audiocaps.github.io/),
[Clotho](https://github.com/audio-captioning/clotho-dataset), and the
[WavCaps](https://github.com/XinhaoMei/WavCaps) collection, which provides the
AudioSet-SL, BBC Sound Effects, and FreeSound subsets.

Download each dataset so that it contains a directory of raw video or audio
files, then set `DATASET` in `cfg/base.yaml` to the root that holds them. The
exact per-dataset sub-paths are listed in `cfg/data.yaml`; edit them if your
layout differs. Any dataset you do not have can simply be commented out there.

### 2. TSV index files

Each split is indexed by a `.tsv` file (one row per clip, keyed by an `id`
column) read from the directory given by `TSV` in `cfg/base.yaml` (default
`./tsv`). These index files are **not** included here. Build them so the `id`
column matches your downloaded filenames; the file name expected for each split
is referenced in `cfg/data.yaml` (e.g. `vgg-train.tsv`, `audiocaps-train.tsv`,
`freesound_corrected.tsv`, ...).

### 3. Extract latents and features

To keep the training loop off the raw-video path, all conditioning features and
VAE latents are pre-extracted once into `.h5` files. Set `H5` in `cfg/base.yaml`
to where they should be written, then:

```bash
torchrun --nproc-per-node 8 latent/extract_audio.py
torchrun --nproc-per-node 8 latent/extract_video.py
```

This is by far the largest artifact (hundreds of GB for the full mix). After
this step the training loop reads only these `.h5` files.

## Training

Set a run name with `exp_id` in `cfg/base.yaml` (output goes to `./out/<exp_id>`,
and `cfg/` is backed up there on the first run), then:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

Training resumes automatically if the run directory already holds a checkpoint;
a post-hoc EMA checkpoint is synthesized at the end. To resume or train into a
custom location, pass `--rundir /path/to/run` instead, having first copied the
configs there (`mkdir -p /path/to/run/cfg && cp cfg/*.yaml /path/to/run/cfg/`),
since `--rundir` reads its configs from that directory rather than `./cfg`.

Hyperparameters live in `cfg/train.yaml` (batch size, number of iterations,
learning-rate schedule, the velocity/consistency split, etc.). If a checkpoint
fails to load after a crash, replace it with the `*_ckpt_shadow.pth` backup
written next to it and retry.

## Evaluation

### 1. Set up the metric toolkit

Quantitative metrics are computed with
[av-benchmark](https://github.com/hkchengrex/av-benchmark). Install it once:

```bash
git clone https://github.com/hkchengrex/av-benchmark.git
cd av-benchmark && pip install -e . && cd ..
```

and fetch the two backbone weights it needs:

```bash
python util/download.py --av-benchmark ./av-benchmark
```

For the VGGSound test set, download the precomputed ground-truth cache
`vggsound-test-eval-cache` from
[Hugging Face](https://huggingface.co/datasets/hkchengrex/MMAudio-precomputed-results)
and point the `gt_cache` field in `cfg/data.yaml` at it. With these in place,
`train.py` reports metrics periodically during training, controlled by
`output_interval.eval` in `cfg/train.yaml`.

### 2. Generate audio for the evaluation set

Set `weight_path` (a `*_ema_final.pth` or `*_last.pth`) and `output_name` in
`cfg/eval.yaml`, choose the number of sampling steps via `cfg/base.yaml: NFE`
(e.g. `1` or `4`), then:

```bash
torchrun --nproc-per-node 8 eval.py
```

The generated `.flac` files are written next to the checkpoint, under
`<run>/<dataset>-<output_name>/`.

### 3. Score the generated audio

Run av-benchmark over the generated clips against the ground-truth cache:

```bash
cd av-benchmark
python evaluate.py \
    --gt_audio   /path/to/groundtruth_audio \
    --gt_cache   /path/to/vggsound-test-eval-cache \
    --pred_audio /path/to/generated_audio \
    --audio_length 8 \
    --num_workers 16 \
    --skip_clap
```

## Single-clip demo

`--video` is optional; without it, FoleyDash runs in text-to-audio mode. Use
`--num-steps` to trade quality for speed (try `1` for the fastest setting).

```bash
python demo.py --duration 8 --num-steps 4 \
    --model-path /path/to/checkpoint.pth \
    --prompt "playing drums" --video /path/to/clip.mp4
```

## Acknowledgements

Built on [MMAudio](https://github.com/hkchengrex/MMAudio). The external
components (VAE, BigVGAN, Synchformer) retain their original licenses, included
alongside their code under `src/ext/`.
