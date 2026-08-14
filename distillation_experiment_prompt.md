# Context Prompt for Claude Code

Copy everything below into Claude Code as the initial message for this project.

---

## Project: Validate the Edge-Device Distillation Pipeline for Subdomain-Compressed Embeddings

### Background

I'm extending a paper called "Subdomain-aware representation compression for pretrained
image embeddings" (Niamluang & Fakcharoenphol, 2026). The paper's core finding: applying
PCA (unsupervised) or LDA (supervised) to pretrained image embeddings (DINOv2, CLIP),
*fit only on a sampled subdomain*, produces compressed embeddings that often **improve**
downstream k-Means clustering quality (V-measure, NMI, ARI, Accuracy) compared to using
the full embedding — even at aggressive compression ratios (down to 5-25% of original
dimension in several datasets).

The paper's motivating use case (Section I-A) is an **edge-device / low-connectivity
scenario**:
1. A user with limited/no ongoing access to the full embedding model collects raw images
   and gets their full embeddings once, remotely.
2. They fit a compression transform (PCA or LDA, using sample embeddings ± labels) to get
   projection matrix W.
3. They store only compressed embeddings locally, and **critically**: they train a small
   local encoder ("student") to approximate raw image → compressed embedding directly,
   so remote access to the big foundation model can be dropped entirely after this point.
4. All future inference (e.g. clustering new images) happens fully locally via the student.

**The paper describes this pipeline as motivation but never implements or evaluates it.**
That's the gap I want to fill: does the distilled local encoder actually preserve the
clustering-quality benefits of the oracle (foundation model → W) pipeline, at a size/
latency budget that's actually edge-appropriate?

### What "oracle" pipeline already exists to reproduce/reuse

- Foundation models: DINOv2 ViT-B/14 (d=768), CLIP ViT-B/32 (d=512), both frozen,
  pretrained, loaded via `timm` / `open_clip` or HuggingFace.
- Datasets: ImageNet-10, ImageNet-Dog-15 (15 dog breed subset of ImageNet-1k),
  TinyImageNet. Also a domain-transfer setup: source domains Dog/Bird/Household/Vehicle
  (~2000 samples each, curated by WordNet synset ID ranges) → target domain Cat
  (~500 samples, 13 classes).
- Compression: PCA (sklearn) and LDA (sklearn), fit on m=5000 sampled embeddings (PCA)
  or fewer for LDA (k bounded by num_classes - 1). Projection matrix W ∈ R^(d×k).
- Evaluation: k-Means (and MiniBatchKMeans) clustering on compressed embeddings,
  scored with V-measure, NMI, ARI, and clustering Accuracy (Hungarian algorithm
  optimal label assignment — `scipy.optimize.linear_sum_assignment`).
- Key reported results to treat as ground truth targets/baselines:
  - Full-embedding baselines and best PCA/LDA k per dataset are in Tables I (CLIP) and
    II (DINOv2) of the paper.
  - Figures 1-4 show score vs. dimension k curves — reproduce a similar sweep if time
    allows, for comparison.

### New experiment to implement: Distillation Pipeline

**Stage 1 (reuse/reproduce oracle).** For a chosen dataset (start with ImageNet-Dog-15,
CLIP backbone — it shows some of the largest gains and lowest compression ratios in the
paper) and chosen method (do both PCA and LDA), fit W on m sampled images to produce
oracle compressed embeddings e' = eW for all images.

**Stage 2 (new: train student encoder).** Train a small CNN `g: image -> R^k` to regress
directly onto e'ᵢ = eᵢW, using ONLY the same m images used to fit W (this matters — the
whole point is no further remote access is needed after this stage).
- Loss: weighted combination of MSE and (1 - cosine similarity), since clustering is
  more sensitive to direction than magnitude. Make the weighting configurable.
- Try 2-3 student architectures spanning a size range appropriate for edge deployment:
  e.g., a MobileNetV3-Small (torchvision, pretrained ImageNet init is fine as a starting
  point since we're not claiming zero-shot), an EfficientNet-Lite0 equivalent, and a
  small custom CNN (~1-5M params) trained from scratch, for comparison.
- Log training/val loss, embedding fidelity (cosine sim + MSE vs oracle e') on a held-out
  split of the m images.

**Stage 3 (evaluation).** On the REST of the dataset (the n images not used for fitting
W or training the student — same held-out split the paper uses for oracle evaluation),
compute three sets of embeddings:
  a. Oracle: full foundation model embedding → W (upper bound; needs "remote" access)
  b. Student: g(image) directly (no foundation model needed at inference)
  c. Random-init student (sanity floor, untrained same architecture)

Run k-Means clustering (k = num true classes, average over 5 seeds, matching paper's
convention) on all three, report V-measure/NMI/ARI/ACC for each.

**Key derived metric — degradation ratio:**
`retained_gain = (student_metric - full_embedding_baseline_metric) / (oracle_metric - full_embedding_baseline_metric)`
This expresses what fraction of the oracle's *improvement over the uncompressed
baseline* survives distillation. This is the number that actually answers "was this
edge pipeline worth building."

**Efficiency measurements per student architecture:**
- Parameter count and model size in MB (fp32 and int8-quantized if easy via
  `torch.quantization`)
- FLOPs (use `thop` or `fvcore`)
- CPU inference latency per image, batch=1, averaged over N runs (simulate edge
  constraint by forcing `torch.set_num_threads(1)` or similar; don't assume GPU
  available at inference time even if training used one)

### Sweeps / ablations, in priority order

1. **PCA target vs LDA target** — does the student distill LDA's more structured,
   lower-dim target better or worse than PCA's variance-maximizing target?
2. **k (compressed dimension)** — repeat across a few k values (reuse k values near
   the paper's reported "best" k per Tables I/II) to see if the retained_gain metric
   degrades at very low k.
3. **m (number of samples available for distillation)** — this is the most important
   ablation for the "practicality" claim. Sweep m down (e.g. 5000, 2000, 1000, 500,
   200) and see where the student collapses. Report retained_gain vs m as the
   headline result of the whole experiment.
4. **Domain transfer robustness** — reuse the paper's Dog→Cat transfer setup (Section
   V-D): fit W and train the student both on the Dog source domain, then evaluate
   clustering on Cat target-domain images run through the student (never seen by W
   or by the student during training). Compare against the oracle's zero-shot
   transfer numbers in Figure 5/6.
5. (If time allows) One-stage (image → e' directly) vs two-stage (image → e via a
   student that approximates the full embedding, then apply W locally) distillation.
   Two-stage needs W stored on-device too, which is cheap (just a matrix), and might
   distill more stably since the target is richer/less compressed. Worth comparing.

### Pre-registered success threshold

Before looking at results, we're calling the edge pipeline "validated" if: the student
retains ≥90% of the oracle's clustering-quality gain over the full-embedding baseline
(averaged across ImageNet-Dog-15 and one other dataset), at <10MB quantized model size
and <50ms CPU latency per image. State this threshold in the writeup up front and report
against it honestly even if it's not met — a clean negative result (e.g., "retains only
60% at m=5000, and collapses below m=1000") is a legitimate and useful finding here.

### Deliverables

1. Clean, modular Python code (data loading, embedding extraction w/ caching to disk
   since foundation-model forward passes are expensive, PCA/LDA fitting, student model
   definitions, training loop, clustering eval, plotting) — not a single notebook dump.
2. Cached embeddings/checkpoints saved so experiments are resumable and re-plottable
   without recomputation.
3. Results tables in the same format as the paper's Tables I/II, but with an added
   "Student" row alongside "Full" and best "PCA"/"LDA" rows.
4. A retained_gain vs. m plot (the headline result).
5. A model size/latency vs. retained_gain Pareto plot across the student architectures.
6. A short markdown summary of findings written against the pre-registered threshold
   above, including honest discussion of any failure modes (e.g., which architecture/
   method combos didn't distill well, and any hypothesis for why).

### Practical / environment notes

**Compute setup: two environments, plan for both.**
- **Heavy workloads (foundation model embedding extraction, student network training,
  full sweeps) run on VSCode-Colab** — assume a single Colab GPU (T4/A100 class, don't
  assume which; detect at runtime via `torch.cuda.get_device_name()` and log it) with
  a session that can disconnect/reset, so:
  - Checkpoint and cache aggressively and often (every N steps/epochs, not just at the
    end) — assume the runtime can vanish mid-run.
  - Write all caches/checkpoints/results to a path that's easy to sync back (e.g. a
    Google Drive-mounted directory, or structure output so it's a simple `zip`/`scp`
    away from the local machine) rather than assuming persistent local disk.
  - Keep Colab-side code runnable as plain `.py` scripts callable from a notebook cell
    (`!python train_student.py --config ...`), not logic embedded only in notebook
    cells, so it's portable and diffable.
- **Local machine: MacBook Pro M4 Max, 36GB unified RAM, no CUDA.** Use this for:
  - Fast iteration / debugging on small subsets (tiny sample counts, 1 epoch, one
    architecture) before pushing a config to Colab for the real run.
  - Post-hoc analysis: loading cached embeddings/results, running clustering + metrics
    (these are cheap — sklearn KMeans on cached vectors, not GPU-bound), and all
    plotting/table generation.
  - PyTorch on this machine should target the `mps` backend (`torch.device("mps")`),
    with a CPU fallback path, since not all ops are MPS-supported — wrap device
    selection in a helper that tries MPS, falls back to CPU, and never assumes CUDA
    is available locally.
- **Practical implication for code structure:** separate "compute-heavy" scripts
  (embedding extraction, student training) from "lightweight" scripts (clustering eval,
  metrics, plotting) into different modules/entry points, so the heavy ones run on
  Colab and the lightweight ones run locally against synced-down cached artifacts.
  Config files (dataset, model, k, m, seed, etc.) should be plain YAML/JSON so the same
  config can be handed to either environment without code changes.
- Foundation model forward passes should be cached to disk (e.g. as .npy or .pt tensors
  keyed by image id) so they're computed once on Colab, then synced down and reused —
  never recomputed per-experiment-run or recomputed locally.
- Use ImageNet-1k subsets via torchvision/HuggingFace `datasets` if full ImageNet isn't
  locally available; TinyImageNet is downloadable directly. Flag clearly if any dataset
  needs to be substituted or subsampled further for compute reasons, and note the
  substitution in the results writeup rather than silently changing the setup.
- Use the same 5-seed averaging convention as the paper for all reported metrics.
- Start with a small pilot (ImageNet-Dog-15, CLIP, PCA only, one student architecture,
  m=2000) run end-to-end on Colab, then sync results down and confirm the pipeline and
  results look reasonable with me on the local machine before scaling up to the full
  sweep matrix.

### First task

Please start by proposing a directory/module structure and a minimal pilot script
(Stage 1 + Stage 2 + Stage 3 for the single pilot config above), and confirm the plan
with me before writing the full sweep infrastructure.
