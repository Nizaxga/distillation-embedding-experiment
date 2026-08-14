"""Chains Stage 1 (oracle fit) -> Stage 2 (student train) -> Stage 3 (eval) for one config.

Usage: python scripts/run_pilot.py --config configs/pilot_dog15_clip_pca.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.compress.fit import fit_pca, fit_lda, save_compression, transform
from src.data.imagenet_dog15 import load_dog15, split_fit_eval
from src.device import get_device
from src.embeddings.cache import load_embeddings
from src.embeddings.extract import extract_clip_embeddings
from src.eval.clustering import cluster_and_score, retained_gain
from src.eval.efficiency import cpu_latency_ms, flops, model_size_mb, param_count
from src.students.models import build_student
from src.students.train import student_input_transform, train_student
from src.utils.config import load_config

from PIL import Image


def print_table(results: dict) -> None:
    metrics = ["v_measure", "nmi", "ari", "acc"]
    header = f"{'row':<14}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    for row_name, scores in results.items():
        print(f"{row_name:<14}" + "".join(f"{scores[m]:>12.4f}" for m in metrics))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()

    print(f"[pilot] run_name={cfg.run_name}")

    # --- data ---
    records = load_dog15(seed=cfg.seed)
    fit_records, eval_records = split_fit_eval(records, cfg.m, seed=cfg.seed)
    print(f"[pilot] fit set={len(fit_records)} eval set={len(eval_records)}")

    # --- Stage 1a: embedding extraction (cached) ---
    extract_clip_embeddings(records, cfg.dataset, cfg.backbone, device=device)

    fit_ids = [r.image_id for r in fit_records]
    eval_ids = [r.image_id for r in eval_records]
    fit_full = load_embeddings(cfg.dataset, cfg.backbone, fit_ids)
    eval_full = load_embeddings(cfg.dataset, cfg.backbone, eval_ids)
    eval_labels = np.array([r.label for r in eval_records])

    # --- Stage 1b: fit compression on fit set only ---
    mean = fit_full.mean(axis=0)
    if cfg.method == "pca":
        W, _ = fit_pca(fit_full - mean, cfg.k, seed=cfg.seed)
    elif cfg.method == "lda":
        fit_labels = np.array([r.label for r in fit_records])
        W, _ = fit_lda(fit_full - mean, fit_labels, cfg.k)
    else:
        raise ValueError(cfg.method)
    save_compression(cfg.run_name, W, mean, cfg.method)

    fit_targets = transform(fit_full, W, mean)  # e' for the m images, student's regression target
    eval_oracle = transform(eval_full, W, mean)  # oracle embeddings for held-out eval

    # --- Stage 2: train student on fit set ONLY ---
    model = train_student(cfg, fit_records, fit_targets, device=device)

    # random-init sanity floor: same architecture, no training
    random_model = build_student(cfg.student_arch, cfg.k)

    # --- Stage 3: compute student + random-init embeddings on eval set ---
    tfm = student_input_transform()

    def embed_with(m: torch.nn.Module) -> np.ndarray:
        m = m.to(device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(eval_records), 64):
                batch = eval_records[i : i + 64]
                imgs = torch.stack(
                    [tfm(Image.open(r.path).convert("RGB")) for r in batch]
                ).to(device)
                out.append(m(imgs).cpu().numpy())
        return np.concatenate(out)

    eval_student = embed_with(model)
    eval_random = embed_with(random_model)

    # --- clustering eval, all rows ---
    n_clusters = len(set(r.label for r in records))
    results = {
        "full": cluster_and_score(eval_full, eval_labels, n_clusters, cfg.eval_seeds),
        f"oracle_{cfg.method}": cluster_and_score(eval_oracle, eval_labels, n_clusters, cfg.eval_seeds),
        "student": cluster_and_score(eval_student, eval_labels, n_clusters, cfg.eval_seeds),
        "random_init": cluster_and_score(eval_random, eval_labels, n_clusters, cfg.eval_seeds),
    }
    print_table(results)

    print("\n[pilot] retained_gain (student / oracle improvement over full):")
    for metric in ["v_measure", "nmi", "ari", "acc"]:
        rg = retained_gain(
            results["student"][metric], results["full"][metric], results[f"oracle_{cfg.method}"][metric]
        )
        print(f"  {metric}: {rg:.4f}")

    # --- efficiency ---
    print("\n[pilot] efficiency:")
    print(f"  params: {param_count(model):,}")
    print(f"  size fp32: {model_size_mb(model):.2f} MB")
    print(f"  size int8: {model_size_mb(model, quantized=True):.2f} MB")
    print(f"  cpu latency: {cpu_latency_ms(model):.2f} ms/image")
    try:
        print(f"  flops: {flops(model)/1e6:.2f} MFLOPs")
    except Exception as e:
        print(f"  flops: failed ({e})")


if __name__ == "__main__":
    main()
