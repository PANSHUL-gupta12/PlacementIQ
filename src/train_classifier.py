"""
Train Classifier — Multi-Model Training & Auto-Selection
==========================================================

Trains five classification models and automatically selects the best one
based on ROC AUC score:
  1. Random Forest
  2. XGBoost
  3. Gradient Boosting
  4. Logistic Regression
  5. Decision Tree

The best model is saved via Joblib to ``models/``.
"""

import os
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed — skipping XGBClassifier")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════
def get_model_registry(random_state: int = 42) -> dict[str, Any]:
    """Return a dictionary of model name → instantiated estimator.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Ordered dict of model name to estimator.
    """
    models: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            random_state=random_state,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    return models


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """Train all models, evaluate, and return results.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        random_state: Random seed.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dictionary with:
          - ``results``: List of dicts with model metrics
          - ``best_model_name``: Name of best model
          - ``best_model``: Fitted best estimator
          - ``all_models``: Dict of name → fitted model
    """
    models = get_model_registry(random_state)
    results: list[dict] = []
    fitted_models: dict[str, Any] = {}

    logger.info("=" * 60)
    logger.info("TRAINING %d CLASSIFIERS", len(models))
    logger.info("=" * 60)

    for name, model in models.items():
        logger.info("Training %s...", name)
        try:
            model.fit(X_train, y_train)
            fitted_models[name] = model

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Test ROC AUC
            test_auc = roc_auc_score(y_test, y_proba)

            # Cross-validation AUC
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring="roc_auc", n_jobs=-1
            )

            result = {
                "model_name": name,
                "test_roc_auc": round(test_auc, 4),
                "cv_roc_auc_mean": round(cv_scores.mean(), 4),
                "cv_roc_auc_std": round(cv_scores.std(), 4),
            }
            results.append(result)

            logger.info("  %s — Test AUC: %.4f | CV AUC: %.4f ± %.4f",
                        name, test_auc, cv_scores.mean(), cv_scores.std())

        except Exception as e:
            logger.error("  Failed to train %s: %s", name, str(e))
            continue

    # ── Select best model ───────────────────────────────────────────────
    if not results:
        raise RuntimeError("No models were successfully trained.")

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model_name"]
    best_model = fitted_models[best_name]

    logger.info("-" * 60)
    logger.info("🏆 Best model: %s (AUC = %.4f)", best_name, results_df.iloc[0]["test_roc_auc"])
    logger.info("-" * 60)

    return {
        "results": results,
        "results_df": results_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "all_models": fitted_models,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Save Model
# ═══════════════════════════════════════════════════════════════════════════
def save_best_model(model: Any, model_name: str, model_dir: str) -> str:
    """Save the best model via Joblib.

    Args:
        model: Fitted estimator.
        model_name: Name identifier.
        model_dir: Directory to save.

    Returns:
        Full path to saved model.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "best_classifier.joblib")
    joblib.dump(model, path)

    # Also save model metadata
    meta = {"model_name": model_name, "model_type": type(model).__name__}
    joblib.dump(meta, os.path.join(model_dir, "classifier_meta.joblib"))

    logger.info("Best model (%s) saved to %s", model_name, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_dataset, detect_columns
    from preprocessing import preprocess_pipeline
    from feature_engineering import feature_engineering_pipeline

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "dataset", "placement.csv")
    out_dir = os.path.join(project_root, "outputs")
    mdl_dir = os.path.join(project_root, "models")

    df = load_dataset(csv_path)
    df = feature_engineering_pipeline(df)
    col_info = detect_columns(df)
    result = preprocess_pipeline(df, col_info, out_dir, mdl_dir)

    training = train_all_models(
        result["X_train"], result["y_train"],
        result["X_test"], result["y_test"],
    )
    save_best_model(training["best_model"], training["best_model_name"], mdl_dir)

    print(f"\n✅ Training complete — Best: {training['best_model_name']}")
    print(training["results_df"].to_string(index=False))
