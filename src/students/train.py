"""HEAVY: student training loop with MSE + (1 - cosine) loss, checkpointed every N steps."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.imagenet_dog15 import ImageRecord
from src.device import get_device
from src.students.models import build_student
from src.utils.config import Config, outputs_root

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def student_input_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DistillDataset(Dataset):
    def __init__(self, records: list[ImageRecord], targets: np.ndarray, transform: transforms.Compose):
        assert len(records) == len(targets)
        self.records = records
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = self.transform(Image.open(r.path).convert("RGB"))
        target = torch.from_numpy(self.targets[idx].astype(np.float32))
        return img, target


def distill_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float, cosine_weight: float):
    mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss = mse_weight * mse + cosine_weight * (1 - cos_sim)
    return loss, mse.item(), cos_sim.item()


def checkpoint_dir(run_name: str) -> Path:
    d = outputs_root() / "checkpoints" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_student(
    cfg: Config,
    fit_records: list[ImageRecord],
    targets: np.ndarray,
    val_frac: float = 0.1,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Train student on (fit_records, targets) ONLY — no images beyond the m used to fit W."""
    device = device or get_device()
    rng = random.Random(cfg.seed)
    idx = list(range(len(fit_records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    tfm = student_input_transform()
    train_ds = DistillDataset([fit_records[i] for i in train_idx], targets[train_idx], tfm)
    val_ds = DistillDataset([fit_records[i] for i in val_idx], targets[val_idx], tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = build_student(cfg.student_arch, cfg.k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ckpt_dir = checkpoint_dir(cfg.run_name)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for imgs, tgt in tqdm(train_loader, desc=f"epoch {epoch} train"):
            imgs, tgt = imgs.to(device), tgt.to(device)
            pred = model(imgs)
            loss, mse, cos = distill_loss(pred, tgt, cfg.mse_weight, cfg.cosine_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            step += 1
            if step % cfg.checkpoint_every_steps == 0:
                torch.save(model.state_dict(), ckpt_dir / f"step_{step}.pt")

        model.eval()
        val_mse, val_cos = [], []
        with torch.no_grad():
            for imgs, tgt in val_loader:
                imgs, tgt = imgs.to(device), tgt.to(device)
                pred = model(imgs)
                _, mse, cos = distill_loss(pred, tgt, cfg.mse_weight, cfg.cosine_weight)
                val_mse.append(mse)
                val_cos.append(cos)
        print(
            f"[train] epoch {epoch}: train_loss={np.mean(train_losses):.4f} "
            f"val_mse={np.mean(val_mse):.4f} val_cos_sim={np.mean(val_cos):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
        scheduler.step()

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    return model
