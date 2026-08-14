# Known Issues, Placeholders, Deferred Work

Honest status list. Nothing below is hidden in the code — grep for the referenced
functions/comments to verify each item yourself.

## Unverified — highest priority to check first

- **The pilot has never been run end-to-end.** No torch was installed in the
  environment this was built in, and dataset access (local ImageNet or HF gated auth)
  wasn't available either. Every module was written to be correct by inspection and
  syntax-checked (`python3 -m py_compile`), but there is zero runtime evidence yet.
  Run it before trusting anything else in these docs about behavior.
- **`k=32` in `configs/pilot_dog15_clip_pca.yaml` is a placeholder.** The plan called
  for using the paper's reported best-k for ImageNet-Dog-15/CLIP/PCA (Table I), but
  that exact value wasn't confirmed against the paper text during this build. Check
  the paper before treating pilot results as comparable to its reported numbers, and
  update this config field.
- **HF fallback's synset-to-label-index matching is fragile**
  (`src/data/imagenet_dog15.py`, `_load_from_hf`, the `name.startswith(syn) or syn in
  name` check). HF's `imagenet-1k` class names may or may not be prefixed with the
  synset id depending on dataset version — this was written from general knowledge of
  the dataset's typical label format, not verified against the actual current HF
  dataset card. If the HF fallback path returns 0 or a wrong count of images, this is
  the first place to look.

## Deferred by design (per the approved pilot plan)

These were explicitly scoped out of the pilot to keep it minimal — see
`../distillation_experiment_prompt.md` and the "What's deferred" section of the plan
that was approved before implementation:

- **Student architectures**: `mobilenetv3_small`, `efficientnet_lite0` — stubbed in
  `src/students/models.py::build_student()`, raise `NotImplementedError`. Only
  `small_cnn` works.
- **LDA path**: `fit_lda()` in `src/compress/fit.py` is implemented and should work,
  but has not been exercised by `run_pilot.py` on real data (pilot config uses
  `method: pca`). Untested in practice.
- **Sweeps**: k-sweep, m-sweep (5000→200, the "headline" ablation per the spec), PCA-
  vs-LDA comparison, domain transfer (Dog→Cat), one-stage vs two-stage distillation —
  none are scripted. Running any of them today means manually editing/copying config
  files and re-invoking `run_pilot.py` per point, then hand-aggregating results. No
  batch-sweep driver script exists yet.
- **Standalone per-stage scripts**: the original structure proposed
  `scripts/extract_embeddings.py`, `scripts/fit_compression.py`,
  `scripts/train_student.py`, `scripts/run_eval.py` as separate entrypoints (so
  lightweight re-eval could run locally without pulling in training code). Only the
  combined `scripts/run_pilot.py` was written. Splitting it is mechanical — the
  functions it calls (`extract_clip_embeddings`, `fit_pca`/`fit_lda`, `train_student`,
  `cluster_and_score`) are already separated in `src/`, `run_pilot.py` just calls them
  in sequence in one process.
- **Plotting**: no retained_gain-vs-m plot, no size/latency Pareto plot, no
  paper-format result tables beyond the plain-text table `run_pilot.py` prints.
- **`requirements.txt` is unpinned** (no version numbers) — fine for a first run, but
  means two people running this at different times could get different library
  versions and non-reproducible results. Pin versions once the pilot is confirmed
  working, so the working version set is captured.

## Not implemented / not fully handled

- **No automatic resume from a checkpoint.** `train_student()`
  (`src/students/train.py`) always starts from a fresh model and epoch 0. If a Colab
  session dies mid-run, the checkpoint files exist on disk but nothing loads them back
  in automatically — you'd need to hand-write a small loader (`model.load_state_dict(
  torch.load(...))`) and adjust the epoch loop to skip completed epochs.
- **No substitution path for ImageNet-Dog-15 if you have neither a local copy nor HF
  access.** The spec mentioned TinyImageNet as a possible substitute dataset if
  ImageNet access is a blocker; this was not implemented — `load_dog15()` will simply
  fail if both data paths are unavailable.
- **FLOPs measurement (`src/eval/efficiency.py::flops`) wraps `thop.profile` in a
  try/except** in `run_pilot.py` because `thop`'s compatibility with arbitrary model
  architectures (especially anything with adaptive pooling) can be inconsistent across
  versions — if it fails, the script logs the error and continues rather than crashing,
  but the FLOPs number will be missing from that run's output.
- **Domain-transfer data (Dog source domains / Cat target domain, Section V-D in the
  paper)** has no loader at all yet — would need a module parallel to
  `imagenet_dog15.py` for the WordNet-synset-range-curated Dog/Bird/Household/Vehicle →
  Cat setup.

## Where to pick this up

Priority order if continuing this work:

1. Run the pilot on Colab, fix whatever breaks (most likely candidates: HF label
   matching, `thop` FLOPs call, or a shape mismatch somewhere in the student training
   loop — none of these paths have executed yet).
2. Confirm `oracle_pca` row roughly matches the paper's reported number for this
   dataset/backbone/method — this validates the data/embedding/PCA path independent of
   anything student-related.
3. Confirm `k=32` against the actual paper table and update the config.
4. Only then move on to sweeps / additional architectures / domain transfer.
