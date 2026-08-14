from pathlib import Path
import numpy as np

from src.utils.config import outputs_root


def embedding_cache_dir(dataset: str, backbone: str) -> Path:
    d = outputs_root() / "embeddings" / dataset / backbone
    d.mkdir(parents=True, exist_ok=True)
    return d


def embedding_path(dataset: str, backbone: str, image_id: str) -> Path:
    return embedding_cache_dir(dataset, backbone) / f"{image_id}.npy"


def has_cached(dataset: str, backbone: str, image_id: str) -> bool:
    return embedding_path(dataset, backbone, image_id).exists()


def save_embedding(dataset: str, backbone: str, image_id: str, vec: np.ndarray) -> None:
    np.save(embedding_path(dataset, backbone, image_id), vec.astype(np.float32))


def load_embedding(dataset: str, backbone: str, image_id: str) -> np.ndarray:
    return np.load(embedding_path(dataset, backbone, image_id))


def load_embeddings(dataset: str, backbone: str, image_ids: list[str]) -> np.ndarray:
    return np.stack([load_embedding(dataset, backbone, iid) for iid in image_ids])
