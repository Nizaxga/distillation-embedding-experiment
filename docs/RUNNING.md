# Running the Pilot

## Prerequisites

Complete [SETUP.md](SETUP.md) first: deps installed, and either `IMAGENET_ROOT` set
to a local ImageNet copy, or HF auth done for the gated-dataset fallback.

## Run the pilot end-to-end

```bash
cd distrillation-model-expr
python scripts/run_pilot.py --config configs/pilot_dog15_clip_pca.yaml
```

Run this **on Colab** (or any machine with a GPU/MPS device and dataset access) — it
does the full CLIP forward pass over ~2000+ images and trains a CNN, both of which are
"HEAVY" per the project's compute split.

### What happens, in order

1. **Load data** — `load_dog15()` pulls the 15-synset image list (local or HF
   fallback), then `split_fit_eval(m=2000)` deterministically splits into the 2000
   "fit" images (used for W + student training) and the remaining "eval" images
   (held out, used only for Stage 3 scoring).
2. **Extract embeddings** — `extract_clip_embeddings()` runs CLIP ViT-B/32 over every
   image once, caching each to `outputs/embeddings/imagenet_dog15/clip_vit_b32/<id>.npy`.
   Re-running the script later skips any image already cached — this step becomes a
   no-op on repeat runs (only the first run or new data pays this cost).
3. **Fit PCA** — on the fit set's embeddings only, producing `W` and `mean`, saved to
   `outputs/compression/<run_name>/`.
4. **Train student** — `SmallCNN` regresses image → oracle-compressed embedding, on the
   fit set only, for `epochs` epochs. Progress bars print per epoch; checkpoints land in
   `outputs/checkpoints/<run_name>/`.
5. **Stage 3 eval** — computes oracle / student / random-init embeddings on the held-out
   eval set, runs KMeans clustering (k = 15, the true class count) with all four
   metrics, and prints:
   - A table: rows = `full`, `oracle_pca`, `student`, `random_init`; columns =
     `v_measure`, `nmi`, `ari`, `acc`.
   - `retained_gain` per metric — the fraction of oracle's improvement-over-full that
     the student preserves. This is the number the pre-registered threshold
     (see `../distillation_experiment_prompt.md`, ≥0.90 average) is judged against.
   - Efficiency block: param count, fp32/int8 size in MB, CPU batch=1 latency in ms,
     FLOPs (best-effort; failures are caught and logged, not fatal).

### Expected runtime (rough, T4-class GPU)

- Embedding extraction: a few minutes for ~2000 images (one-time, cached after).
- Student training: single-digit minutes for 10 epochs on `small_cnn` at `m=2000`.
- Everything else (PCA fit, clustering, efficiency) is seconds.

If a Colab session disconnects mid-training, the most recent checkpoint under
`outputs/checkpoints/<run_name>/` survives (if that directory is on a Drive mount or
otherwise persisted) — automatic resume from a checkpoint is not wired up yet, see
KNOWN_ISSUES.md.

## Interpreting the output

- **`full` row** is the uncompressed-CLIP-embedding clustering baseline.
- **`oracle_pca` row** is the paper's original pipeline result (foundation model → W).
  Sanity-check this against the paper's Table I ImageNet-Dog-15/CLIP/PCA number before
  trusting anything downstream — if this doesn't roughly match, something in data
  loading, embedding extraction, or PCA fitting is off, and the student numbers are
  meaningless until it's fixed.
- **`student` row** is the fully-local pipeline this whole experiment is validating.
- **`random_init` row** is the sanity floor — student minus all training.
- **`retained_gain`** near 1.0 = student preserves oracle's full improvement over full
  embeddings. Near 0 = student is no better than the uncompressed baseline. Negative =
  student is worse than not compressing at all.

## Syncing results back to local machine

Zip and download (or Drive-sync) the whole `outputs/` directory. It's structured so a
plain copy is sufficient — no path rewriting needed, everything is keyed by
dataset/backbone/run_name/image_id, all relative to `outputs/`.

## What's NOT runnable yet

- `scripts/extract_embeddings.py`, `scripts/fit_compression.py`,
  `scripts/train_student.py`, `scripts/run_eval.py` — standalone per-stage entrypoints
  described in the original plan. Only the combined `run_pilot.py` exists today. See
  KNOWN_ISSUES.md.
- Any student architecture other than `small_cnn`.
- Any sweep (k, m, method comparison, domain transfer) — only the single pilot config
  has been wired up end-to-end. Sweeps require iterating configs and are not yet
  scripted as a batch.
