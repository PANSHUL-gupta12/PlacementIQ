"""
Unsupervised Learning — Clustering & Cluster Evaluation
=========================================================

Covers the complete unsupervised learning curriculum:
  1. Clustering Concepts & Distance Metrics
  2. K-Means Clustering with Elbow Method & Silhouette Analysis
  3. Hierarchical Clustering (Agglomerative) with Dendrogram Analysis
  4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  5. Multi-Metric Evaluation:
     - Silhouette Score
     - Davies-Bouldin Index
     - Calinski-Harabasz Index
  6. Real-World Applications: Student Placement Readiness Segmentation
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ═══════════════════════════════════════════════════════════════════════════
# 1. K-Means Clustering & Optimal K Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_kmeans_range(
    X: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
    subsample_size: int = 5000,
) -> Dict[str, Any]:
    """Evaluate K-Means across a range of clusters using Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz."""
    if len(X) > subsample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X), subsample_size, replace=False)
        X_eval = X[indices]
    else:
        X_eval = X

    inertias, silhouettes, db_indices, ch_indices = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X_eval)
        inertias.append(km.inertia_)
        sil = float(silhouette_score(X_eval, labels))
        silhouettes.append(sil)
        db = float(davies_bouldin_score(X_eval, labels))
        db_indices.append(db)
        ch = float(calinski_harabasz_score(X_eval, labels))
        ch_indices.append(ch)
        logger.info("K=%d | Inertia: %.2f | Silhouette: %.4f | DB: %.4f | CH: %.2f", k, km.inertia_, sil, db, ch)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_bouldin": db_indices,
        "calinski_harabasz": ch_indices,
        "best_k": best_k,
    }


def plot_cluster_evaluation_metrics(k_results: Dict[str, Any], output_dir: str) -> str:
    """Plot Elbow Inertia, Silhouette Score, and Davies-Bouldin Score side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ks = k_results["k_range"]

    # 1. Elbow
    axes[0].plot(ks, k_results["inertias"], "bo-", linewidth=2)
    axes[0].set_title("Elbow Method (Inertia / WCSS)", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].axvline(k_results["best_k"], color="red", linestyle="--", label=f"Best K={k_results['best_k']}")
    axes[0].legend()

    # 2. Silhouette
    axes[1].plot(ks, k_results["silhouettes"], "go-", linewidth=2)
    axes[1].set_title("Silhouette Score (Higher is Better)", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].axvline(k_results["best_k"], color="red", linestyle="--", label=f"Best K={k_results['best_k']}")
    axes[1].legend()

    # 3. Davies-Bouldin
    axes[2].plot(ks, k_results["davies_bouldin"], "mo-", linewidth=2)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "unsupervised_cluster_evaluation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hierarchical Clustering & Dendrogram
# ═══════════════════════════════════════════════════════════════════════════
def run_hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = 4,
    linkage_method: str = "ward",
    sample_size: int = 500,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Run Agglomerative Hierarchical Clustering and generate a dendrogram."""
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X[sample_indices]

    # Dendrogram
    dendro_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        Z = linkage(X_sample, method=linkage_method)
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., leaf_font_size=10., ax=ax)
        ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method.title()} Linkage)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Cluster Sample index / Cluster size")
        ax.set_ylabel("Distance")
        dendro_path = os.path.join(output_dir, "hierarchical_dendrogram.png")
        fig.savefig(dendro_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage_method)
    labels = model.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels)
    logger.info("Hierarchical Clustering (%s) Silhouette Score: %.4f", linkage_method, sil)
    return labels, dendro_path


# ═══════════════════════════════════════════════════════════════════════════
# 3. DBSCAN (Density-Based Spatial Clustering)
# ═══════════════════════════════════════════════════════════════════════════
def run_dbscan_clustering(
    X: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 15,
    sample_size: int = 4000,
) -> Dict[str, Any]:
    """Run DBSCAN to identify core, border points, and outliers/noise."""
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X[sample_indices]

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_sample)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    sil = None
    if n_clusters > 1:
        mask = labels != -1
        if sum(mask) > n_clusters:
            sil = float(silhouette_score(X_sample[mask], labels[mask]))

    logger.info("DBSCAN: %d clusters, %d noise points (%.2f%%)", n_clusters, n_noise, noise_ratio * 100)
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "silhouette": sil,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Applications of Clustering: Student Profile Segmentation
# ═══════════════════════════════════════════════════════════════════════════
def segment_students(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """Analyze cluster profiles for student placement readiness segmentation."""
    df_copy = df_features.copy()
    if len(labels) < len(df_copy):
        df_copy = df_copy.iloc[:len(labels)].copy()
    df_copy["Cluster"] = labels

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    summary = df_copy.groupby("Cluster")[numeric_cols].mean().round(3)

    summary_path = os.path.join(output_dir, "student_cluster_profiles.csv")
    summary.to_csv(summary_path)
    logger.info("Student cluster segment profiles saved to: %s", summary_path)
    return summary

