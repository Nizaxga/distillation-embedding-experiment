from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Config:
    dataset: str
    backbone: str
    method: str          # "pca" or "lda"
    k: int
    m: int
    seed: int
    student_arch: str
    mse_weight: float
    cosine_weight: float
    epochs: int
    batch_size: int
    lr: float
    eval_seeds: list = field(default_factory=lambda: [0])
    checkpoint_every_steps: int = 200
    run_name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.run_name:
            self.run_name = (
                f"{self.dataset}_{self.backbone}_{self.method}_k{self.k}_m{self.m}"
                f"_{self.student_arch}_seed{self.seed}"
            )


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    known = {k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
    extra = {k: v for k, v in raw.items() if k not in Config.__dataclass_fields__}
    return Config(**known, extra=extra)


def outputs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs"
