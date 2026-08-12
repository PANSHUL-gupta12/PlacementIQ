"""
Train Cluster — KMeans Clustering with Auto-K Selection
=========================================================

Trains KMeans clustering on student features to group them into
meaningful clusters. Automatically selects the optimal K using:
  1. Elbow Method (inertia)
  2. Silhouette Score

Generates cluster names based on centroid analysis.
Saves model and visualizations.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Cluster Name Templates (ordered by overall strength descending)
# ---------------------------------------------------------------------------
CLUSTER_NAME_POOL: list[str] = [
    "Top Performers",
    "Strong Coders",
    "Academic Achievers",
    "Well-Rounded Students",
    "Developing Talent",
    "Needs Improvement",
    "At Risk",
    "Beginners",
]


# ═══════════════════════════════════════════════════════════════════════════
# Optimal K Selection
# ═══════════════════════════════════════════════════════════════════════════
def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict:
    """Evaluate K values via inertia and silhouette score.

    Args:
        X: Feature matrix (scaled).
        k_range: Range of K values to test.
        random_state: Random seed.

    Returns:
        Dict with ``inertias``, ``silhouettes``, ``best_k``.
    """
    inertias: list[float] = []
    silhouettes: list[float] = []

    logger.info("Evaluating K from %d to %d...", k_range.start, k_range.stop - 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        logger.info("  K=%d — Inertia: %.2f | Silhouette: %.4f", k, km.inertia_, sil)

    best_k = k_range.start + int(np.argmax(silhouettes))
    logger.info("Best K by silhouette: %d (score=%.4f)", best_k, max(silhouettes))

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════
def plot_elbow_and_silhouette(k_info: dict, output_dir: str) -> None:
    """Save elbow and silhouette plots side by side.

    Args:
        k_info: Output from ``find_optimal_k()``.
        output_dir: Directory to save plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(k_info["k_range"], k_info["inertias"], "bo-", linewidth=2, markersize=8)
    ax1.axvline(x=k_info["best_k"], color="red", linestyle="--", alpha=0.7,
                label=f"Best K = {k_info['best_k']}")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2.plot(k_info["k_range"], k_info["silhouettes"], "go-", linewidth=2, markersize=8)
    ax2.axvline(x=k_info["best_k"], color="red", linestyle="--", alpha=0.7,
                label=f"Best K = {k_info['best_k']}")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimal Cluster Selection", fontsize=16, fontweight="bold")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cluster_elbow_silhouette.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Elbow & silhouette plot saved to %s", path)


def plot_cluster_distribution(labels: np.ndarray, cluster_names: dict, output_dir: str) -> None:
    """Save cluster distribution bar chart.

    Args:
        labels: Cluster labels for each sample.
        cluster_names: Dict mapping cluster ID → name.
        output_dir: Directory to save.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    names = [cluster_names.get(i, f"Cluster {i}") for i in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.bar(names, counts.values, color=colors, edgecolor="black")

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", fontweight="bold")

    ax.set_title("Student Cluster Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cluster_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Cluster distribution saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Naming
# ═══════════════════════════════════════════════════════════════════════════
def assign_cluster_names(
    centroids: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """Assign meaningful names based on classifier prediction probability.

    Clusters are ranked by their predicted placement probability
    and mapped to names from the pool.

    Args:
        centroids: KMeans cluster centres, shape (K, n_features).
        feature_names: Feature column names.

    Returns:
        Dict mapping cluster_id → human-readable name.
    """
    import os
    import joblib
    try:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        clf = joblib.load(os.path.join(model_dir, "best_classifier.joblib"))
        probs = clf.predict_proba(centroids)[:, 1]
    except Exception as e:
        logger.warning(f"Could not load classifier to name clusters. Falling back to simple sum. Error: {e}")
        probs = centroids.mean(axis=1)

    k = centroids.shape[0]
    ranked_indices = np.argsort(-probs)  # Descending

    if k <= 2:
        names = ["Top Performers", "Needs Improvement"]
    else:
        # Spread names across the pool based on K
        step = max(1, len(CLUSTER_NAME_POOL) // k)
        names = [CLUSTER_NAME_POOL[i * step] for i in range(k)]

    cluster_names = {int(ranked_indices[i]): names[i] for i in range(k)}
    logger.info("Cluster names: %s", cluster_names)
    return cluster_names


# ═══════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def train_cluster_pipeline(
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    model_dir: str,
    random_state: int = 42,
) -> dict:
    """Train KMeans with auto-K, assign names, save model & plots.

    Args:
        X: Scaled feature matrix.
        feature_names: Feature column names.
        output_dir: Directory for visualizations.
        model_dir: Directory for model persistence.
        random_state: Random seed.

    Returns:
        Dict with ``model``, ``labels``, ``cluster_names``, ``best_k``.
    """
    logger.info("=" * 60)
    logger.info("CLUSTERING PIPELINE START")
    logger.info("=" * 60)

    # 1. Find optimal K
    k_info = find_optimal_k(X, random_state=random_state)
    plot_elbow_and_silhouette(k_info, output_dir)

    # 2. Train final model
    best_k = k_info["best_k"]
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    # 3. Assign names
    cluster_names = assign_cluster_names(km.cluster_centers_, feature_names)

    # 4. Visualize
    plot_cluster_distribution(labels, cluster_names, output_dir)

    # 5. Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(km, os.path.join(model_dir, "cluster_model.joblib"))
    joblib.dump(cluster_names, os.path.join(model_dir, "cluster_names.joblib"))
    logger.info("Cluster model saved to %s", model_dir)

    logger.info("=" * 60)
    logger.info("CLUSTERING PIPELINE COMPLETE — K=%d", best_k)
    logger.info("=" * 60)

    return {
        "model": km,
        "labels": labels,
        "cluster_names": cluster_names,
        "best_k": best_k,
        "k_info": k_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Predict Cluster for New Data
# ═══════════════════════════════════════════════════════════════════════════
def predict_cluster(
    X: np.ndarray,
    model: Optional[KMeans] = None,
    cluster_names: Optional[dict] = None,
    model_dir: Optional[str] = None,
) -> dict:
    """Predict cluster for new student data.

    Args:
        X: Feature vector(s), shape (1, n_features) or (n, n_features).
        model: Fitted KMeans model (or load from model_dir).
        cluster_names: Cluster name mapping (or load from model_dir).
        model_dir: Directory with saved models.

    Returns:
        Dict with ``cluster_id`` and ``cluster_name``.
    """
    if model is None and model_dir:
        model = joblib.load(os.path.join(model_dir, "cluster_model.joblib"))
    if cluster_names is None and model_dir:
        cluster_names = joblib.load(os.path.join(model_dir, "cluster_names.joblib"))

    if model is None:
        raise ValueError("No model provided or loadable.")

    cluster_id = int(model.predict(X.reshape(1, -1) if X.ndim == 1 else X)[0])
    name = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"

    return {"cluster_id": cluster_id, "cluster_name": name}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Use via predict.py for the full pipeline.")
