from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.utils.config import outputs_root


def compression_dir(run_name: str) -> Path:
    d = outputs_root() / "compression" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_pca(embeddings: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(embeddings)
    W = pca.components_.T  # d x k, so e' = (e - mean) @ W
    return W, pca


def fit_lda(embeddings: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, LDA]:
    n_classes = len(set(labels.tolist()))
    max_k = n_classes - 1
    if k > max_k:
        raise ValueError(f"LDA k={k} exceeds num_classes-1={max_k}")
    lda = LDA(n_components=k)
    lda.fit(embeddings, labels)
    W = lda.scalings_[:, :k]
    return W, lda


def transform(embeddings: np.ndarray, W: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    x = embeddings - mean if mean is not None else embeddings
    return x @ W


def save_compression(run_name: str, W: np.ndarray, mean: np.ndarray, method: str) -> None:
    d = compression_dir(run_name)
    np.save(d / "W.npy", W)
    np.save(d / "mean.npy", mean)
    (d / "method.txt").write_text(method)


def load_compression(run_name: str) -> tuple[np.ndarray, np.ndarray]:
    d = compression_dir(run_name)
    return np.load(d / "W.npy"), np.load(d / "mean.npy")
