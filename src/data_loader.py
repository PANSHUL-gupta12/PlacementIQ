"""
Data Loader — Dataset Loading & Exploratory Data Analysis
==========================================================

Loads the placement CSV dataset, auto-detects column types and the target
column, and generates a comprehensive set of EDA visualizations.

All plots are saved to the ``outputs/`` directory.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the placement CSV and log basic info.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded dataset from %s — shape %s", filepath, df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Column Detection
# ═══════════════════════════════════════════════════════════════════════════
def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect categorical, numerical, and target columns.

    Heuristic for target: binary integer column whose name contains common
    placement keywords, or the last binary column.

    Args:
        df: The raw DataFrame.

    Returns:
        Dictionary with keys ``categorical``, ``numerical``, ``target``.
    """
    target_keywords = ["placed", "status", "placement", "target", "label", "hired"]
    target_col: Optional[str] = None

    # Try keyword match first
    for col in df.columns:
        if col.lower().replace("_", "") in [k.replace("_", "") for k in target_keywords]:
            target_col = col
            break
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break

    # Fallback: last binary column
    if target_col is None:
        for col in reversed(df.columns.tolist()):
            if df[col].dropna().nunique() == 2:
                target_col = col
                break

    if target_col is None:
        raise ValueError("Could not auto-detect target column. Please specify manually.")

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from feature lists
    numerical = [c for c in numerical if c != target_col]
    categorical = [c for c in categorical if c != target_col]
    
    # Also remove Student_ID and Salary_LPA if they exist, since they are not classification features
    for non_feat in ["Student_ID", "Salary_LPA"]:
        if non_feat in numerical:
            numerical.remove(non_feat)
        if non_feat in categorical:
            categorical.remove(non_feat)

    info = {
        "categorical": categorical,
        "numerical": numerical,
        "target": target_col,
    }
    logger.info("Detected columns:\n  Target    : %s\n  Numerical : %s\n  Categorical: %s",
                target_col, numerical, categorical)
    return info


# ═══════════════════════════════════════════════════════════════════════════
# Missing Value Report
# ═══════════════════════════════════════════════════════════════════════════
def missing_value_report(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate and save a missing-value report.

    Args:
        df: The DataFrame.
        output_dir: Directory to save the report CSV.

    Returns:
        DataFrame with missing-value statistics.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing": missing, "Percent": pct})
    report = report[report["Missing"] > 0].sort_values("Percent", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "missing_value_report.csv")
    report.to_csv(report_path)
    logger.info("Missing value report saved to %s", report_path)

    if report.empty:
        logger.info("No missing values detected.")
    else:
        logger.info("Missing value summary:\n%s", report.to_string())

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Feature Summary
# ═══════════════════════════════════════════════════════════════════════════
def feature_summary(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate descriptive statistics and save as CSV.

    Args:
        df: The DataFrame.
        output_dir: Directory to save the summary.

    Returns:
        DataFrame with describe() output.
    """
    summary = df.describe(include="all").T
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "feature_summary.csv")
    summary.to_csv(summary_path)
    logger.info("Feature summary saved to %s", summary_path)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Save a correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numerical columns for a heatmap.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Correlation heatmap saved to %s", path)


def plot_class_distribution(df: pd.DataFrame, target_col: str, output_dir: str) -> None:
    """Save a bar chart of target-class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"] if len(counts) == 2 else sns.color_palette("muted")
    counts.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
    ax.set_title(f"Class Distribution — {target_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")

    # Annotate bars
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")

    path = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Class distribution plot saved to %s", path)


def plot_histograms(df: pd.DataFrame, numerical_cols: list[str], output_dir: str) -> None:
    """Save histograms for all numerical features."""
    n_cols = len(numerical_cols)
    if n_cols == 0:
        return

    n_plot_cols = 3
    n_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5 * n_plot_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=25, color="#3498db", edgecolor="white", alpha=0.85)
        axes[idx].set_title(col, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Histograms", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "histograms.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Histograms saved to %s", path)


def plot_boxplots(df: pd.DataFrame, numerical_cols: list[str], output_dir: str) -> None:
    """Save boxplots for all numerical features."""
    n_cols = len(numerical_cols)
    if n_cols == 0:
        return

    n_plot_cols = 3
    n_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5 * n_plot_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[idx], color="#9b59b6", width=0.4)
        axes[idx].set_title(col, fontsize=11, fontweight="bold")

    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Boxplots", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "boxplots.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Boxplots saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_eda(filepath: str, output_dir: str) -> tuple[pd.DataFrame, dict]:
    """Run full EDA pipeline: load, detect, report, visualise.

    Args:
        filepath: Path to placement CSV.
        output_dir: Directory for outputs.

    Returns:
        Tuple of (DataFrame, column_info dict).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(filepath)
    col_info = detect_columns(df)

    # Reports
    missing_value_report(df, output_dir)
    feature_summary(df, output_dir)

    # Visualizations
    plot_correlation_heatmap(df, output_dir)
    plot_class_distribution(df, col_info["target"], output_dir)
    plot_histograms(df, col_info["numerical"], output_dir)
    plot_boxplots(df, col_info["numerical"], output_dir)

    logger.info("EDA complete — all outputs in %s", output_dir)
    return df, col_info


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "dataset", "train.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "dataset", "cleaned_train.csv")
    out_dir = os.path.join(project_root, "outputs")
    df, info = run_eda(csv_path, out_dir)
    print(f"\n✅ EDA complete — {len(df)} records analysed")
    print(f"   Target column: {info['target']}")
    print(f"   Visualizations saved to: {out_dir}")
