"""
Evaluate Classifier — Comprehensive Model Evaluation
======================================================

Evaluates trained classification models with:
  - Accuracy, Precision, Recall, F1, ROC AUC
  - Confusion Matrix
  - ROC Curve comparison plot
  - Model comparison bar chart

All visualizations are saved to ``outputs/``.
"""

import os
import logging
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ═══════════════════════════════════════════════════════════════════════════
# Single Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a single model and return all metrics.

    Args:
        model: Fitted estimator.
        model_name: Name for logging.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dict with all evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    logger.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# All Models Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_all_models(
    all_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame.

    Args:
        all_models: Dict of model_name → fitted estimator.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        DataFrame with evaluation metrics for each model.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING %d MODELS", len(all_models))
    logger.info("=" * 60)

    results = []
    for name, model in all_models.items():
        metrics = evaluate_model(model, name, X_test, y_test)
        results.append(metrics)

    # Build comparison table (exclude array columns)
    table_cols = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comparison_df = pd.DataFrame([{k: r[k] for k in table_cols} for r in results])
    comparison_df = comparison_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    logger.info("\nModel Comparison:\n%s", comparison_df.to_string(index=False))
    return comparison_df


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        model_name: Model identifier.
        output_dir: Directory to save.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Placed", "Placed"],
        yticklabels=["Not Placed", "Placed"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower()}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def plot_roc_curves(
    all_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Plot ROC curves for all models on a single figure.

    Args:
        all_models: Dict of model_name → fitted estimator.
        X_test: Test features.
        y_test: True labels.
        output_dir: Directory to save.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("husl", len(all_models))

    for (name, model), color in zip(all_models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "roc_curves_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("ROC curves saved to %s", path)


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """Save a grouped bar chart comparing model metrics.

    Args:
        comparison_df: DataFrame with model metrics.
        output_dir: Directory to save.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    melted = comparison_df.melt(
        id_vars="model_name", value_vars=metrics,
        var_name="Metric", value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=melted, x="model_name", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", path)


def save_comparison_csv(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """Save the model comparison table as CSV.

    Args:
        comparison_df: DataFrame with model metrics.
        output_dir: Directory to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(path, index=False)
    logger.info("Model comparison CSV saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def evaluation_pipeline(
    all_models: dict[str, Any],
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
) -> pd.DataFrame:
    """Run the complete evaluation pipeline.

    Args:
        all_models: Dict of model_name → fitted estimator.
        best_model_name: Name of the best model.
        X_test: Test features.
        y_test: True labels.
        output_dir: Output directory.

    Returns:
        Comparison DataFrame.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Evaluate all
    comparison_df = evaluate_all_models(all_models, X_test, y_test)

    # Confusion matrix for best model
    best_model = all_models[best_model_name]
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_model_name, output_dir)

    # ROC curves
    plot_roc_curves(all_models, X_test, y_test, output_dir)

    # Comparison chart
    plot_model_comparison(comparison_df, output_dir)

    # Save CSV
    save_comparison_csv(comparison_df, output_dir)

    # Classification report for best model
    report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])
    logger.info("\nBest Model (%s) Classification Report:\n%s", best_model_name, report)

    return comparison_df


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Use via train_classifier.py or predict.py")
