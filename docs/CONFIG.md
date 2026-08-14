# Config Reference

Configs are YAML, loaded by `load_config()` in `src/utils/config.py` into a `Config`
dataclass. Same file works on Colab or local — nothing environment-specific lives in
config (that's `IMAGENET_ROOT` / device detection, both env/runtime-level).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `dataset` | str | Dataset identifier, e.g. `imagenet_dog15`. Used as a cache-path key. |
| `backbone` | str | Foundation model identifier, e.g. `clip_vit_b32`. Cache-path key. |
| `method` | str | `pca` or `lda` — compression method. |
| `k` | int | Compressed dimension. For LDA, must be ≤ `num_classes - 1`. |
| `m` | int | Number of images used to fit W AND train the student. Central ablation variable. |
| `seed` | int | Controls the fit/eval split, PCA random_state, and student training shuffle/init. |
| `student_arch` | str | `small_cnn` (implemented). `mobilenetv3_small` / `efficientnet_lite0` raise `NotImplementedError` — deferred. |
| `mse_weight` | float | Weight on MSE term in distillation loss. |
| `cosine_weight` | float | Weight on `(1 - cosine_similarity)` term. |
| `epochs` | int | Student training epochs. |
| `batch_size` | int | Student training batch size. |
| `lr` | float | AdamW learning rate for student training. |
| `eval_seeds` | list[int] | Seeds averaged over in clustering eval (paper convention: 5; pilot default: `[0]`). |
| `checkpoint_every_steps` | int | Save a student checkpoint every N optimizer steps, in addition to every epoch. |
| `run_name` | str | Optional override. If empty, auto-derived: `{dataset}_{backbone}_{method}_k{k}_m{m}_{student_arch}_seed{seed}`. This is the key used for `outputs/compression/<run_name>/` and `outputs/checkpoints/<run_name>/`. |
| *(anything else)* | — | Passed through into `Config.extra` (dict), not validated. Use for ablation-specific fields not yet promoted to first-class (e.g. domain-transfer source/target dataset names). |

## Pilot config values (`configs/pilot_dog15_clip_pca.yaml`)

```yaml
dataset: imagenet_dog15
backbone: clip_vit_b32
method: pca
k: 32              # placeholder — see KNOWN_ISSUES.md
m: 2000
seed: 0
student_arch: small_cnn
mse_weight: 0.5
cosine_weight: 0.5
epochs: 10
batch_size: 64
lr: 0.0003
eval_seeds: [0]
checkpoint_every_steps: 50
```

To run a different config, copy this file, edit fields, point `--config` at the new
path. No code changes needed for any of: k, m, seed, method (pca/lda), loss weights,
epochs, batch size, lr. Changing `student_arch` to anything but `small_cnn` will fail
until those architectures are implemented (KNOWN_ISSUES.md).
