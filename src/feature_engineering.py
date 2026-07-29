"""
Feature Engineering — Composite Score Construction
=====================================================

Creates derived features from raw student data to improve model performance.
All engineered features are normalized to [0, 1].

Composite scores:
  - AcademicScore: CGPA + Attendance - Backlogs
  - TechnicalScore: CodingScore + Projects + GitHub + Hackathons
  - ExperienceScore: Internship + Certifications + Semester
  - CommunicationScore: Communication + LeadershipScore
  - ProjectScore: Projects + Hackathons + GitHub
  - CodingIndex: CodingScore + Aptitude
  - OverallSkillScore: Weighted combination of all composites
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions — maps score name to (column list, weights)
# ---------------------------------------------------------------------------
COMPOSITE_FEATURES: dict[str, dict] = {
    "AcademicScore": {
        "columns": ["CGPA", "Backlogs"],
        "weights": [0.6, -0.4],
    },
    "TechnicalScore": {
        "columns": ["Coding_Skills", "Projects", "Aptitude_Test_Score"],
        "weights": [0.50, 0.30, 0.20],
    },
    "ExperienceScore": {
        "columns": ["Internships", "Certifications", "Age"],
        "weights": [0.50, 0.30, 0.20],
    },
    "CommunicationScore": {
        "columns": ["Communication_Skills", "Soft_Skills_Rating"],
        "weights": [0.60, 0.40],
    },
    "ProjectScore": {
        "columns": ["Projects"],
        "weights": [1.0],
    },
    "CodingIndex": {
        "columns": ["Coding_Skills", "Aptitude_Test_Score"],
        "weights": [0.60, 0.40],
    },
}

OVERALL_WEIGHTS: dict[str, float] = {
    "AcademicScore": 0.25,
    "TechnicalScore": 0.25,
    "ExperienceScore": 0.15,
    "CommunicationScore": 0.15,
    "ProjectScore": 0.10,
    "CodingIndex": 0.10,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _safe_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1], handling constant columns."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


# ═══════════════════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════════════════
def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all composite features from raw columns.

    Missing source columns are skipped gracefully.

    Args:
        df: DataFrame with raw (unscaled) features.

    Returns:
        DataFrame with added composite feature columns.
    """
    logger.info("Creating composite features...")

    for feat_name, spec in COMPOSITE_FEATURES.items():
        cols = spec["columns"]
        weights = spec["weights"]

        # Filter to columns that exist in the DataFrame
        available = [(c, w) for c, w in zip(cols, weights) if c in df.columns]
        if not available:
            logger.warning("Skipping %s — no source columns found", feat_name)
            continue

        avail_cols, avail_weights = zip(*available)

        # Normalise each contributing column before combining
        normalised = pd.DataFrame({
            c: _safe_normalize(df[c].astype(float)) for c in avail_cols
        })

        # Weighted sum
        score = sum(w * normalised[c] for c, w in zip(avail_cols, avail_weights))
        df[feat_name] = _safe_normalize(score).round(4)
        logger.info("  Created %s from %s", feat_name, list(avail_cols))

    return df


def create_overall_skill_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create the OverallSkillScore as a weighted combination of composites.

    Args:
        df: DataFrame that must already have composite features.

    Returns:
        DataFrame with ``OverallSkillScore`` column added.
    """
    available = {k: v for k, v in OVERALL_WEIGHTS.items() if k in df.columns}
    if not available:
        logger.warning("No composite features found — skipping OverallSkillScore")
        return df

    total_weight = sum(available.values())
    score = sum(
        (w / total_weight) * df[col] for col, w in available.items()
    )
    df["OverallSkillScore"] = _safe_normalize(score).round(4)
    logger.info("  Created OverallSkillScore from %s", list(available.keys()))
    return df


def normalize_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final MinMax normalization pass on all engineered features.

    Args:
        df: DataFrame with composite features.

    Returns:
        DataFrame with normalized composite features.
    """
    engineered_cols = list(COMPOSITE_FEATURES.keys()) + ["OverallSkillScore"]
    existing = [c for c in engineered_cols if c in df.columns]

    if existing:
        scaler = MinMaxScaler()
        df[existing] = scaler.fit_transform(df[existing])
        logger.info("Normalised %d engineered features to [0, 1]", len(existing))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        df: Raw (or cleaned, unscaled) DataFrame.

    Returns:
        DataFrame with all engineered features added and normalized.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE START")
    logger.info("=" * 60)

    df = create_composite_features(df)
    df = create_overall_skill_score(df)
    df = normalize_engineered_features(df)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
    logger.info("=" * 60)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    from data_loader import load_dataset

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "dataset", "placement.csv")

    df = load_dataset(csv_path)
    df = feature_engineering_pipeline(df)

    print(f"\n✅ Feature engineering complete")
    print(f"   New columns: {[c for c in df.columns if c in list(COMPOSITE_FEATURES.keys()) + ['OverallSkillScore']]}")
    print(df.head())
