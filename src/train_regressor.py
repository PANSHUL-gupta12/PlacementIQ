"""
Train Regressor — Linear, Multiple Linear & Polynomial Regression
===================================================================

Covers Supervised Learning - Regression for Continuous Evaluation:
  1. Regression Concepts and Applications (Campus Salary/CTC in LPA)
  2. Simple Linear Regression
  3. Multiple Linear Regression
  4. Polynomial Regression (Degree 2 & 3)
  5. Regression Performance Metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Coefficient of Determination (R² Score)
  6. Residual Analysis & Model Comparison
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
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Regression Metrics Calculation
# ═══════════════════════════════════════════════════════════════════════════
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, MSE, RMSE, and R² Score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Model Training & Comparison
# ═══════════════════════════════════════════════════════════════════════════
def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs",
    model_dir: str = "models",
    save_models: bool = True,
    scaled_input: bool = False,
) -> Dict[str, Any]:
    """Train Simple Linear, Multiple Linear, and Polynomial Regression models."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    results = []
    fitted_models = {}
    predictions = {}

    # 1. Simple Linear Regression (Using top predictor: Coding_Skills / CGPA)
    primary_feature = "Coding_Skills" if "Coding_Skills" in X_train.columns else X_train.columns[0]
    simple_lr = LinearRegression()
    simple_lr.fit(X_train[[primary_feature]], y_train)
    y_pred_slr = simple_lr.predict(X_test[[primary_feature]])
    metrics_slr = calculate_regression_metrics(y_test, y_pred_slr)
    results.append({
        "Model": f"Simple Linear Regression ({primary_feature})",
        **metrics_slr,
    })
    fitted_models["SimpleLinearRegression"] = simple_lr
    predictions["Simple Linear Regression"] = y_pred_slr

    # 2. Multiple Linear Regression (All input features)
    mult_lr = LinearRegression()
    mult_lr.fit(X_train, y_train)
    y_pred_mlr = mult_lr.predict(X_test)
    metrics_mlr = calculate_regression_metrics(y_test, y_pred_mlr)
    results.append({
        "Model": "Multiple Linear Regression",
        **metrics_mlr,
    })
    fitted_models["MultipleLinearRegression"] = mult_lr
    predictions["Multiple Linear Regression"] = y_pred_mlr

    # 3. Polynomial Regression (Degree 2)
    poly2_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    poly2_pipe.fit(X_train, y_train)
    y_pred_poly2 = poly2_pipe.predict(X_test)
    metrics_poly2 = calculate_regression_metrics(y_test, y_pred_poly2)
    results.append({
        "Model": "Polynomial Regression (Degree 2)",
        **metrics_poly2,
    })
    fitted_models["PolynomialRegression_Deg2"] = poly2_pipe
    predictions["Polynomial Regression (Deg 2)"] = y_pred_poly2

    # Results Table
    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    table_path = os.path.join(output_dir, "regression_metrics_comparison.csv")
    results_df.to_csv(table_path, index=False)
    logger.info("Regression comparison:\n%s", results_df.to_string(index=False))

    # Best model selection
    best_name = results_df.iloc[0]["Model"]
    best_key = "MultipleLinearRegression" if "Multiple" in best_name else "PolynomialRegression_Deg2"
    best_model = fitted_models[best_key]

    # Save best regressor
    if save_models:
        joblib.dump(best_model, os.path.join(model_dir, "best_regressor.joblib"))
        joblib.dump(
            {
                "model_name": best_name,
                "metrics": results_df.iloc[0].to_dict(),
                "scaled_input": scaled_input,
            },
            os.path.join(model_dir, "regressor_meta.joblib"),
        )

    # Plot 1: Actual vs Predicted
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (m_name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test[:300], y_pred[:300], alpha=0.5, color="teal", edgecolors="none")
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal Fit (y=x)")
        ax.set_title(m_name, fontweight="bold")
        ax.set_xlabel("Actual Salary (LPA)")
        ax.set_ylabel("Predicted Salary (LPA)")
        ax.legend()
    fig.suptitle("Actual vs. Predicted Salary Packages (LPA)", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    plot_path1 = os.path.join(output_dir, "regression_actual_vs_predicted.png")
    fig.savefig(plot_path1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Residual Analysis
    fig, ax = plt.subplots(figsize=(9, 5))
    residuals = y_test - y_pred_mlr
    sns.histplot(residuals, kde=True, color="purple", ax=ax, bins=30)
    ax.set_title("Residual Distribution (Actual - Predicted LPA) — Multiple Linear Regression", fontweight="bold")
    ax.set_xlabel("Residual Error (LPA)")
    ax.axvline(0, color="red", linestyle="--")
    plot_path2 = os.path.join(output_dir, "regression_residuals.png")
    fig.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "results_df": results_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "models": fitted_models,
        "plot_actual_vs_pred": plot_path1,
        "plot_residuals": plot_path2,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(project_root, "dataset", "train.csv")
    test_csv = os.path.join(project_root, "dataset", "test.csv")
    out_dir = os.path.join(project_root, "outputs")
    mdl_dir = os.path.join(project_root, "models")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    reg_features = [
        "CGPA", "Coding_Skills", "Internships", "Projects",
        "Communication_Skills", "Aptitude_Test_Score", "Soft_Skills_Rating",
        "Certifications", "Backlogs",
    ]
    res = train_regression_models(
        df_train[reg_features], df_train["Salary_LPA"],
        df_test[reg_features], df_test["Salary_LPA"],
        output_dir=out_dir, model_dir=mdl_dir,
        save_models=True, scaled_input=False,
    )
    print(f"\n✅ Regression Training Complete — Best Model: {res['best_model_name']}")
    print(res["results_df"].to_string(index=False))

