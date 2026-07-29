"""
Explain Prediction — Explainable AI with Feature Importance & SHAP
====================================================================

Provides human-readable explanations for placement predictions by
identifying which features increase and decrease placement probability.

Uses:
  1. Feature Importance (tree-based models) — always available
  2. SHAP values (TreeExplainer) — when SHAP is installed

Generates explanations like:
  ✔ CGPA increased prediction
  ✖ No internship reduced prediction
"""

import os
import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed — using feature importance only")

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
# Feature Importance (Always Available)
# ═══════════════════════════════════════════════════════════════════════════
def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importance from a trained model.

    Supports tree-based models (``feature_importances_``) and linear
    models (``coef_``).

    Args:
        model: Fitted estimator.
        feature_names: Feature column names.

    Returns:
        DataFrame with columns ``feature`` and ``importance``, sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model does not expose feature importances.")
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Top 5 features:\n%s", imp_df.head().to_string(index=False))
    return imp_df


def plot_feature_importance(
    imp_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """Save a horizontal bar chart of feature importance.

    Args:
        imp_df: Feature importance DataFrame.
        output_dir: Directory to save.
        top_n: Number of top features to display.
    """
    plot_df = imp_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(plot_df))
    ax.barh(plot_df["feature"], plot_df["importance"], color=colors, edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance — Placement Prediction", fontsize=14, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Explanations
# ═══════════════════════════════════════════════════════════════════════════
def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Compute SHAP values for the given data.

    Args:
        model: Fitted estimator (tree-based preferred).
        X: Feature matrix.
        output_dir: If provided, saves SHAP summary plot.

    Returns:
        SHAP values array, or None if SHAP is unavailable.
    """
    if not HAS_SHAP:
        logger.warning("SHAP not available — skipping.")
        return None

    try:
        logger.info("Computing SHAP values...")

        # Try TreeExplainer first (for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            # Fallback: LinearExplainer for linear models
            try:
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            except Exception:
                # Final fallback: use a small sample with KernelExplainer
                logger.info("Using KernelExplainer (slower)...")
                sample = X.iloc[:50] if len(X) > 50 else X
                explainer = shap.KernelExplainer(model.predict_proba, sample)
                shap_values = explainer.shap_values(sample)
                X = sample  # For the summary plot

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X, show=False)
            path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close("all")
            logger.info("SHAP summary plot saved to %s", path)

        return shap_values

    except Exception as e:
        logger.warning("SHAP computation failed: %s — falling back to feature importance", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Human-Readable Explanation
# ═══════════════════════════════════════════════════════════════════════════
def explain_single_prediction(
    model: Any,
    student_features: np.ndarray,
    feature_names: list[str],
    shap_values: Optional[np.ndarray] = None,
    student_index: int = 0,
) -> dict:
    """Generate a human-readable explanation for a single student.

    Uses SHAP values if available, otherwise falls back to feature
    importance combined with the student's relative feature values.

    Args:
        model: Fitted estimator.
        student_features: Feature vector(s).
        feature_names: Feature column names.
        shap_values: Precomputed SHAP values (optional).
        student_index: Index of the student in the batch.

    Returns:
        Dict with ``positive_factors``, ``negative_factors``, and
        ``explanation_text``.
    """
    if student_features.ndim == 1:
        student_features = student_features.reshape(1, -1)

    # ── Use SHAP if available ───────────────────────────────────────────
    if shap_values is not None:
        if shap_values.ndim == 1:
            sv = shap_values
        else:
            sv = shap_values[student_index]

        contributions = list(zip(feature_names, sv))
    else:
        # Fallback: feature_importance × feature_value as proxy
        imp_df = get_feature_importance(model, feature_names)
        imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))
        fv = student_features[0]

        contributions = [
            (feat, imp_map.get(feat, 0) * fv[i])
            for i, feat in enumerate(feature_names)
        ]

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    positive = [(f, v) for f, v in contributions if v > 0]
    negative = [(f, v) for f, v in contributions if v < 0]

    # ── Build text ──────────────────────────────────────────────────────
    lines = []
    lines.append("📊 Prediction Explanation:")
    lines.append("")

    if positive:
        lines.append("  ✅ Factors INCREASING placement chance:")
        for feat, val in positive[:5]:
            lines.append(f"     ✔ {feat} (contribution: {val:+.4f})")

    if negative:
        lines.append("")
        lines.append("  ⚠️  Factors REDUCING placement chance:")
        for feat, val in negative[:5]:
            lines.append(f"     ✖ {feat} (contribution: {val:+.4f})")

    explanation_text = "\n".join(lines)
    logger.info("\n%s", explanation_text)

    return {
        "positive_factors": positive[:5],
        "negative_factors": negative[:5],
        "explanation_text": explanation_text,
        "all_contributions": contributions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def explanation_pipeline(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: str,
) -> dict:
    """Run the full explanation pipeline.

    Args:
        model: Best fitted model.
        X_test: Test features.
        feature_names: Feature column names.
        output_dir: Directory for outputs.

    Returns:
        Dict with ``feature_importance``, ``shap_values``.
    """
    logger.info("=" * 60)
    logger.info("EXPLAINABLE AI PIPELINE")
    logger.info("=" * 60)

    # Feature importance
    imp_df = get_feature_importance(model, feature_names)
    plot_feature_importance(imp_df, output_dir)

    # SHAP
    shap_vals = compute_shap_values(model, X_test, output_dir)

    return {
        "feature_importance": imp_df,
        "shap_values": shap_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Use via predict.py for the full pipeline.")
