# Setup

## Two environments

This project is split for two machines (see `../distillation_experiment_prompt.md`
"Practical / environment notes" for the full rationale):

- **Colab (GPU)** — runs everything expensive: embedding extraction, student training.
- **Local Mac (M4 Max, MPS, no CUDA)** — fast iteration, and lightweight eval/plotting
  against cached results synced down from Colab.

Config files under `configs/` are plain YAML and work unmodified on either machine.

## Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`: `torch`, `torchvision`, `open_clip_torch`, `timm`,
`scikit-learn`, `scipy`, `numpy`, `pyyaml`, `datasets`, `thop`, `Pillow`, `tqdm`,
`matplotlib`.

On Colab this is a normal `!pip install -r requirements.txt` cell. On the Mac, use a
venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Torch will use MPS automatically on the Mac if available — see `src/device.py`. No
GPU code path assumes CUDA is present; it's CUDA > MPS > CPU with graceful fallback.

## Getting ImageNet-Dog-15 data

The dataset is **15 specific dog-breed synsets** from ImageNet-1k (the standard subset
used in deep-clustering literature — list is in `src/data/imagenet_dog15.py`,
`DOG15_SYNSETS`). There is no separate "ImageNet-Dog-15" download — you get it by
filtering full ImageNet-1k to these 15 synsets. Three supported paths, tried in order:

### Option A — precached dir shared with `colab_transfer_disentangler.ipynb` (preferred)

If a related notebook already downloaded ImageNet-Dog-15 to Google Drive at
`representation-learning/.cache/imagenet-dog-15/<label>/*.jpg` (the same
`download_domain()` layout used by `colab_transfer_disentangler.ipynb` for its own
domains), the loader picks this up automatically once Drive is mounted at
`/content/drive` — no extra config needed. To point at a different location, set:

```bash
export IMAGENET_DOG15_CACHE=/path/to/cache/imagenet-dog-15
```

Label subdirs may be synset ids (`n02085936/`) or plain ILSVRC integer indices —
both are handled (`_load_from_precached_dir` in `src/data/imagenet_dog15.py`). This
is the fastest path: no re-download, no HF gated-dataset auth.

### Option B — local ImageNet copy

If you already have ImageNet-1k on disk in ImageFolder layout
(`<root>/<synset_id>/*.JPEG`, e.g. `<root>/n02085936/xxx.JPEG`), point at it:

```bash
export IMAGENET_ROOT=/path/to/imagenet/train
```

The loader (`load_dog15()` in `src/data/imagenet_dog15.py`) will pull just the 15
synset subdirectories from under this root.

### Option C — HuggingFace `imagenet-1k` fallback (no local copy)

If neither the precached dir nor `IMAGENET_ROOT` is available, the loader falls back
to streaming HF's `imagenet-1k` dataset and filtering to the 15 synsets, caching
matched images to `outputs/raw_images/dog15_hf/`. This path:

- **Requires HF auth** — `imagenet-1k` on HF is gated. Run `huggingface-cli login`
  first (needs a HF account that has accepted the ImageNet-1k terms on the dataset
  page).
- **Is slower** — it's a streaming scan of the full 1.28M-image train split, filtering
  down to the ~1950 images across the 15 target classes as it goes.
- **Prints a loud log line** when triggered (`[data] ... falling back to HF ...`) —
  this is a deliberate substitution flag per the experiment spec ("flag clearly if any
  dataset needs to be substituted... note the substitution in the results writeup").

If you have none of: a precached Drive dir, a local ImageNet copy, or HF access to the
gated dataset, you cannot run this pipeline as-is — this is the one hard external
dependency. (Substitute datasets like TinyImageNet were considered in the spec as a
fallback, but the loader does not currently implement that substitution — see
KNOWN_ISSUES.md.)

## Colab-specific notes

- Detect GPU at runtime: `src/device.py`'s `get_device()` prints
  `torch.cuda.get_device_name(0)` when CUDA is available — check the pilot's first log
  lines to see whether you got a T4 or A100-class instance.
- Colab sessions can disconnect. The student training loop
  (`src/students/train.py`) checkpoints every `checkpoint_every_steps` steps (config
  field) AND at the end of every epoch, to `outputs/checkpoints/<run_name>/`. If a
  session dies mid-epoch, the most recent `step_N.pt` is still there — resuming from it
  is not yet automated (see KNOWN_ISSUES.md), but the file exists to resume from
  manually.
- Point `outputs/` at a Google Drive mount if you want it to survive a Colab reset
  without manual syncing, e.g. by symlinking or by setting the repo's working directory
  under `/content/drive/MyDrive/...`. Nothing in the code hardcodes a path outside the
  repo — `outputs_root()` in `src/utils/config.py` resolves relative to the repo root,
  so moving the whole repo onto a Drive mount is all that's needed.

## Local Mac notes

- No dataset download or training happens locally for the pilot as designed — local
  runs are for lightweight re-eval from *synced-down* caches (`outputs/embeddings/`,
  `outputs/compression/`, `outputs/checkpoints/`). A lightweight `scripts/run_eval.py`
  entrypoint is planned but not yet written (see KNOWN_ISSUES.md) — for now, reuse the
  clustering/efficiency functions in `src/eval/` directly against synced caches.
- To sync down from Colab: zip `outputs/` on Colab, download, unzip into the same path
  locally, or `scp`/Drive-sync it. Directory layout under `outputs/` is stable and
  content-addressed by `run_name` / image id, so this is a plain file copy — no
  transformation needed.
