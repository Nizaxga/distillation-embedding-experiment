import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = dict(zip(row_ind, col_ind))
    mapped_pred = np.array([mapping[p] for p in y_pred])
    return float((mapped_pred == y_true).mean())


def cluster_and_score(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int, seeds: list[int]) -> dict:
    """KMeans + {V-measure, NMI, ARI, ACC}, averaged over seeds (paper's 5-seed convention)."""
    metrics = {"v_measure": [], "nmi": [], "ari": [], "acc": []}
    for seed in seeds:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        pred = km.fit_predict(embeddings)
        metrics["v_measure"].append(v_measure_score(labels, pred))
        metrics["nmi"].append(normalized_mutual_info_score(labels, pred))
        metrics["ari"].append(adjusted_rand_score(labels, pred))
        metrics["acc"].append(clustering_accuracy(labels, pred))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def retained_gain(student_metric: float, full_metric: float, oracle_metric: float) -> float:
    denom = oracle_metric - full_metric
    if abs(denom) < 1e-9:
        return float("nan")
    return (student_metric - full_metric) / denom
