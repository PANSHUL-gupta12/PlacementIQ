"""
Preprocessing — Data Cleaning, Encoding & Scaling
===================================================

Handles the complete preprocessing pipeline:
  1. Duplicate removal
  2. Missing value imputation
  3. Categorical encoding (LabelEncoder)
  4. Outlier detection (IQR method)
  5. Feature scaling (StandardScaler)
  6. Train-test split

All fitted transformers are saved via Joblib for inference reuse.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Duplicate Removal
# ═══════════════════════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        De-duplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (before=%d, after=%d)", removed, before, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Missing Value Handling
# ═══════════════════════════════════════════════════════════════════════════
def handle_missing_values(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Impute missing values: median for numerical, mode for categorical.

    Args:
        df: Input DataFrame.
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.

    Returns:
        DataFrame with imputed values.
    """
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %s with median=%.2f", col, median_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Imputed %s with mode=%s", col, mode_val)

    remaining = df.isnull().sum().sum()
    logger.info("Remaining missing values: %d", remaining)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Categorical Encoding
# ═══════════════════════════════════════════════════════════════════════════
def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode all categorical columns.

    Args:
        df: Input DataFrame.
        categorical_cols: Columns to encode.

    Returns:
        Tuple of (encoded DataFrame, dict of fitted LabelEncoders).
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded %s — classes: %s", col, list(le.classes_))
    return df, encoders


# ═══════════════════════════════════════════════════════════════════════════
# Outlier Detection
# ═══════════════════════════════════════════════════════════════════════════
def detect_outliers(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    """Detect and cap outliers using the IQR method.

    Values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR are clipped.

    Args:
        df: Input DataFrame.
        numerical_cols: Columns to check.

    Returns:
        DataFrame with outliers capped.
    """
    total_capped = 0
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += outliers
            logger.info("Capped %d outliers in %s (range: [%.2f, %.2f])",
                        outliers, col, lower, upper)

    logger.info("Total outliers capped: %d", total_capped)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Feature Scaling
# ═══════════════════════════════════════════════════════════════════════════
def scale_features(
    df: pd.DataFrame,
    numerical_cols: list[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale numerical features.

    Args:
        df: Input DataFrame.
        numerical_cols: Columns to scale.
        scaler: Pre-fitted scaler (for inference). If None, fits a new one.

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info("Fitted and applied StandardScaler to %d columns", len(numerical_cols))
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        logger.info("Applied pre-fitted StandardScaler to %d columns", len(numerical_cols))
    return df, scaler


# ═══════════════════════════════════════════════════════════════════════════
# Save Cleaned Data
# ═══════════════════════════════════════════════════════════════════════════
def save_cleaned_data(df: pd.DataFrame, output_dir: str) -> str:
    """Save the cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        output_dir: Directory to save.

    Returns:
        Full path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(path, index=False)
    logger.info("Cleaned data saved to %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Save Transformers
# ═══════════════════════════════════════════════════════════════════════════
def save_transformers(
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
    model_dir: str,
) -> None:
    """Save scaler and encoders via Joblib.

    Args:
        scaler: Fitted StandardScaler.
        encoders: Dict of fitted LabelEncoders.
        model_dir: Directory to save.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "encoders.joblib"))
    logger.info("Transformers saved to %s", model_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_info: dict,
    output_dir: str,
    model_dir: str,
) -> dict:
    """Run the full preprocessing pipeline on separate train and test datasets.

    Args:
        df_train: Raw train DataFrame.
        df_test: Raw test DataFrame.
        col_info: Column info from ``data_loader.detect_columns()``.
        output_dir: Directory for cleaned data.
        model_dir: Directory for transformers.

    Returns:
        Dictionary with keys: ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``scaler``, ``encoders``, ``feature_names``, ``numerical_cols``,
        ``categorical_cols``.
    """
    target_col = col_info["target"]
    numerical_cols = col_info["numerical"]
    categorical_cols = col_info["categorical"]

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("=" * 60)

    # Step 1: Duplicates (Train only)
    df_train = remove_duplicates(df_train)
    # Test set duplicates are usually left as is, but we can remove them or keep them. Let's keep test rows intact.
    
    # Store sizes to split back later
    n_train = len(df_train)
    
    # Concatenate for consistent encoding and scaling
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Step 2: Missing values
    df = handle_missing_values(df, numerical_cols, categorical_cols)

    # Step 2.5: Target encoding (if target is categorical string)
    if df[target_col].dtype == object or str(df[target_col].dtype) in ["string", "category", "str", "string[python]", "string[pyarrow]"]:
        # Check if 'Placed'/'Not Placed' format
        unique_vals = set(df[target_col].dropna().unique())
        if unique_vals == {"Placed", "Not Placed"}:
            df[target_col] = df[target_col].map({"Placed": 1, "Not Placed": 0})
        else:
            # Fallback to LabelEncoder
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))

    # Step 3: Encode categoricals
    df, encoders = encode_categoricals(df, categorical_cols)
    # After encoding, previously categorical cols are now numerical
    all_numerical = numerical_cols + categorical_cols

    # Step 4: Outlier detection
    df = detect_outliers(df, all_numerical)

    # Step 5: Save cleaned (before scaling)
    save_cleaned_data(df, output_dir)

    # Step 6: Feature scaling
    df_scaled, scaler = scale_features(df.copy(), all_numerical)

    # Step 7: Train-test split (Back to original)
    feature_cols = [c for c in df_scaled.columns if c != target_col and c != "Student_ID"]
    X = df_scaled[feature_cols]
    y = df_scaled[target_col]

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_test = y.iloc[n_train:]
    
    logger.info("Train set: %d, Test set: %d", len(X_train), len(X_test))

    # Step 8: Save transformers
    save_transformers(scaler, encoders, model_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_cols,
        "numerical_cols": all_numerical,
        "categorical_cols": categorical_cols,
        "df_clean": df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_loader import load_dataset, detect_columns

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "dataset", "placement.csv")
    out_dir = os.path.join(project_root, "outputs")
    mdl_dir = os.path.join(project_root, "models")

    df = load_dataset(csv_path)
    col_info = detect_columns(df)
    result = preprocess_pipeline(df, col_info, out_dir, mdl_dir)

    print(f"\n✅ Preprocessing complete")
    print(f"   Train: {result['X_train'].shape}  Test: {result['X_test'].shape}")
    print(f"   Features: {result['feature_names']}")
