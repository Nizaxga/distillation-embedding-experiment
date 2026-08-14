# Architecture

## Directory map

```
distrillation-model-expr/
├── configs/                         # YAML configs, portable across Colab/local
│   └── pilot_dog15_clip_pca.yaml
├── src/
│   ├── device.py                    # CUDA > MPS > CPU fallback; CPU-only helper for latency measurement
│   ├── data/
│   │   └── imagenet_dog15.py        # 15-synset dataset loader (local ImageFolder or HF fallback), fit/eval split
│   ├── embeddings/
│   │   ├── extract.py               # HEAVY: CLIP forward pass, batched
│   │   └── cache.py                 # disk cache read/write, keyed by (dataset, backbone, image_id)
│   ├── compress/
│   │   └── fit.py                   # PCA/LDA fit + transform, W/mean persistence
│   ├── students/
│   │   ├── models.py                # SmallCNN (implemented); mobilenetv3_small/efficientnet_lite0 (stubbed, NotImplementedError)
│   │   └── train.py                 # HEAVY: training loop, MSE+cosine loss, step + epoch checkpointing
│   ├── eval/
│   │   ├── clustering.py            # KMeans + V-measure/NMI/ARI/Hungarian-ACC, seed averaging, retained_gain formula
│   │   └── efficiency.py            # param count, fp32/int8 size, FLOPs (thop), CPU batch=1 latency
│   └── utils/
│       └── config.py                # YAML -> Config dataclass, outputs_root() path helper
├── scripts/
│   └── run_pilot.py                 # orchestrates all 3 stages end-to-end for one config
├── outputs/                         # gitignored; all caches/checkpoints/results, resumable & syncable
└── docs/                            # you are here
```

## Data flow (pilot)

```
load_dog15() ──> split_fit_eval(m) ──> fit_records (m), eval_records (n)
                                            │
                                extract_clip_embeddings()      [cached: outputs/embeddings/<dataset>/<backbone>/<image_id>.npy]
                                            │
                            fit_full (m×768)         eval_full (n×768)
                                    │
                         fit_pca / fit_lda(fit_full) ──> W, mean   [saved: outputs/compression/<run_name>/{W,mean}.npy]
                                    │
                    fit_targets = transform(fit_full, W)     eval_oracle = transform(eval_full, W)
                            │
              train_student(fit_records, fit_targets)   <- ONLY these m images, no further foundation-model access
                            │  [checkpoints: outputs/checkpoints/<run_name>/{step_N,epoch_N,final}.pt]
                        trained model g
                            │
              eval_student = g(eval_records images)     eval_random = random_init_g(eval_records images)
                            │
        cluster_and_score() on {eval_full, eval_oracle, eval_student, eval_random} vs eval_labels
                            │
            retained_gain(student_metric, full_metric, oracle_metric)   <- headline number
```

## Key design decisions

- **Caching is content-addressed, not run-addressed.** Embeddings live at
  `outputs/embeddings/<dataset>/<backbone>/<image_id>.npy` — independent of k, m,
  method, seed. Compute an embedding once, reuse it across every config that shares
  dataset+backbone. Compression (`W`) and checkpoints ARE run-addressed
  (`outputs/compression/<run_name>/`, `outputs/checkpoints/<run_name>/`) since those
  depend on method/k/m/arch/seed — `run_name` encodes all of that (see CONFIG.md).
- **The student never sees more than the `m` fit images.** This is enforced by
  construction in `run_pilot.py` — `train_student()` is only ever called with
  `fit_records`/`fit_targets`. `eval_records` (the `n` held-out images) are used solely
  for Stage 3 inference, never for training. This is the core constraint the whole
  experiment is testing ("no further remote access after fitting W").
- **Efficiency metrics always run on CPU** (`src/eval/efficiency.py`,
  `cpu_latency_ms`), regardless of what device training used — the edge-deployment
  claim is about CPU inference, and `torch.set_num_threads(1)` simulates a
  single-core edge constraint.
- **Random-init student is a genuine sanity floor**, not a mock — same architecture
  (`build_student(arch, k)`), freshly initialized weights, no training. If the trained
  student doesn't clearly beat this on clustering metrics, something is broken.
