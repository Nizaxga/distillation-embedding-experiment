# Docs Index

This documents the edge-device distillation pipeline built to validate the motivating
use case in "Subdomain-aware representation compression for pretrained image
embeddings" (Niamluang & Fakcharoenphol, 2026). The full experiment spec that drove
this build is at `../distillation_experiment_prompt.md` — read that first for the
scientific goal, ablations, and success threshold. These docs cover the *implementation*:
what exists, how to run it, and what's still missing.

- [SETUP.md](SETUP.md) — install deps, get ImageNet-Dog-15 data, Colab vs local setup
- [RUNNING.md](RUNNING.md) — exact commands to run the pilot end-to-end, what each stage produces
- [ARCHITECTURE.md](ARCHITECTURE.md) — module map, data flow, where caches/checkpoints live
- [CONFIG.md](CONFIG.md) — every YAML config field, what it controls
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — placeholders, unverified assumptions, deferred work

## Current state in one paragraph

A pilot-only pipeline exists for **ImageNet-Dog-15 + CLIP ViT-B/32 + PCA + one student
architecture (`small_cnn`)**. It has never been run end-to-end yet (no torch installed
locally as of writing, and it needs either a local ImageNet copy or HF auth for data).
The code is real and complete for the pilot scope — not a stub — but it is unverified.
Treat any numbers it produces as unconfirmed until someone runs `scripts/run_pilot.py`
and sanity-checks the output against the paper's reported ImageNet-Dog-15/CLIP/PCA
numbers.
