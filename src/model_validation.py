"""
Model Evaluation and Validation Module
=======================================

Covers the complete Model Evaluation and Validation syllabus:
  1. Train-Test Split (80/20 Stratified Validation)
  2. Cross-Validation Techniques (Standard K-Fold vs. Stratified K-Fold)
  3. Overfitting and Underfitting Diagnosis (Validation Curves across Tree Depths)
  4. Bias-Variance Trade-off Analysis
  5. Confusion Matrix (TP, TN, FP - Type I, FN - Type II)
  6. Performance Metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
"""

import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, validation_curve, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cross-Validation: K-Fold vs. Stratified K-Fold
# ═══════════════════════════════════════════════════════════════════════════
def compare_cv_techniques(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compare Standard K-Fold vs. Stratified K-Fold cross validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    kf_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
    skf_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    return {
        "kfold_mean": float(kf_scores.mean()),
        "kfold_std": float(kf_scores.std()),
        "stratified_mean": float(skf_scores.mean()),
        "stratified_std": float(skf_scores.std()),
        "kfold_scores": kf_scores.tolist(),
        "stratified_scores": skf_scores.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Overfitting & Underfitting: Validation Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_overfitting_underfitting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str = "outputs",
) -> str:
    """Diagnose Underfitting vs. Overfitting using Decision Tree validation curve."""
    # Subsample for faster curve calculation if large
    if len(X_train) > 5000:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), 5000, replace=False)
        X_sub, y_sub = X_train.iloc[idx], y_train.iloc[idx]
    else:
        X_sub, y_sub = X_train, y_train

    depth_range = np.arange(1, 15)
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X_sub, y_sub,
        param_name="max_depth",
        param_range=depth_range,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depth_range, train_mean, 'o-', color="crimson", label="Training Accuracy", linewidth=2)
    ax.fill_between(depth_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="crimson")

    ax.plot(depth_range, test_mean, 's-', color="royalblue", label="Cross-Validation Accuracy", linewidth=2)
    ax.fill_between(depth_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color="royalblue")

    # Annotate Underfitting, Optimal, and Overfitting regions
    ax.axvspan(1, 3, color='yellow', alpha=0.15, label='Underfitting Zone (High Bias)')
    ax.axvspan(3, 7, color='green', alpha=0.15, label='Optimal Complexity Zone')
    ax.axvspan(7, 14, color='orange', alpha=0.15, label='Overfitting Zone (High Variance)')

    ax.set_title("Model Complexity vs. Performance: Overfitting & Underfitting Diagnosis", fontweight="bold", fontsize=13)
    ax.set_xlabel("Decision Tree Depth (Model Complexity)", fontsize=11)
    ax.set_ylabel("Accuracy Score", fontsize=11)
    ax.legend(loc="lower right")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "overfitting_underfitting_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bias-Variance Trade-off Concept Visualizer
# ═══════════════════════════════════════════════════════════════════════════
def plot_bias_variance_tradeoff(output_dir: str = "outputs") -> str:
    """Plot theoretical Bias-Variance Trade-off decomposition curves."""
    complexity = np.linspace(1, 10, 100)
    bias_sq = 10 / (complexity + 1)
    variance = 0.12 * (complexity ** 1.8)
    irreducible_error = np.full_like(complexity, 0.8)
    total_error = bias_sq + variance + irreducible_error

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(complexity, bias_sq, 'b--', label='Bias² (Underfitting Error)', linewidth=2)
    ax.plot(complexity, variance, 'g--', label='Variance (Overfitting Error)', linewidth=2)
    ax.plot(complexity, irreducible_error, 'k:', label='Irreducible Noise', linewidth=1.5)
    ax.plot(complexity, total_error, 'r-', label='Total Expected Error', linewidth=2.5)

    opt_idx = np.argmin(total_error)
    ax.axvline(complexity[opt_idx], color='purple', linestyle='--', label=f'Optimal Model Complexity')

    ax.set_title("The Bias-Variance Trade-off", fontweight="bold", fontsize=14)
    ax.set_xlabel("Model Complexity", fontsize=11)
    ax.set_ylabel("Error", fontsize=11)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "bias_variance_tradeoff.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 4. Detailed Confusion Matrix Breakdown
# ═══════════════════════════════════════════════════════════════════════════
def generate_confusion_matrix_breakdown(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Compute and visualize detailed Confusion Matrix with TP, TN, FP, FN."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [
        [f"True Negative (TN)\n{tn}\n(Correctly Rejected)", f"False Positive (FP)\n{fp}\n(Type I Error)"],
        [f"False Negative (FN)\n{fn}\n(Type II Error)", f"True Positive (TP)\n{tp}\n(Correctly Placed)"],
    ]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 11, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Actual Label", fontweight="bold")
    ax.set_xticklabels(["Not Placed (0)", "Placed (1)"])
    ax.set_yticklabels(["Not Placed (0)", "Placed (1)"])
    ax.set_title("Confusion Matrix with Error Types", fontweight="bold", fontsize=13)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "detailed_confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "plot_path": out_path,
    }

