"""
Predict — End-to-End PlacementIQ AI Pipeline
===============================================

Orchestrates the complete pipeline:
  1. Generate dataset (if not exists)
  2. Load & EDA
  3. Feature engineering
  4. Preprocessing
  5. Train classifiers (compare 5 models, select best)
  6. Evaluate models (metrics, plots)
  7. Train KMeans clustering
  8. Explainable AI (feature importance + SHAP)
  9. Single-student prediction with full report

Usage:
    python src/predict.py                  # Full pipeline
    python src/predict.py --predict-only   # Skip training, use saved models
"""

import os
import sys
import logging
import argparse

# Fix OpenBLAS memory allocation error on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Ensure src/ is on path for sibling imports
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import run_eda, load_dataset, detect_columns
from preprocessing import preprocess_pipeline
from feature_engineering import feature_engineering_pipeline, COMPOSITE_FEATURES
from train_classifier import train_all_models, save_best_model
from evaluate_classifier import evaluation_pipeline
from train_cluster import train_cluster_pipeline, predict_cluster
from explain_prediction import explanation_pipeline, explain_single_prediction
from recommendation_engine import (
    analyse_skill_gaps,
    generate_roadmap,
    format_skill_gap_text,
    format_roadmap_text,
)
from company_matching import match_companies, format_company_text
from resume_scoring import score_resume, format_resume_text
from interview_scoring import evaluate_interview, format_interview_text
from report_generator import generate_report, save_report, classify_readiness

# LLM module is optional (requires torch + transformers)
try:
    from llm_resume_classifier import generate_resume_text as llm_generate_text
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "cleaned_train.csv")
DATASET_TEST_PATH = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Ensure Dataset Exists
# ═══════════════════════════════════════════════════════════════════════════
def ensure_datasets() -> None:
    """Ensure train and test datasets exist."""
    if os.path.isfile(DATASET_TRAIN_PATH) and os.path.isfile(DATASET_TEST_PATH):
        logger.info("Datasets found.")
        return

    logger.error("Datasets not found in dataset folder. Please provide cleaned_train.csv and test.csv")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Full Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def run_training_pipeline() -> dict:
    """Execute the complete training pipeline.

    Returns:
        Dict with all pipeline artefacts (models, data splits, etc.).
    """
    logger.info("=" * 70)
    logger.info("   PLACEMENTIQ AI — FULL TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Dataset & EDA ───────────────────────────────────────────
    ensure_datasets()
    logger.info("\n▶ STEP 1: Exploratory Data Analysis")
    df_raw_train, col_info_raw = run_eda(DATASET_TRAIN_PATH, OUTPUT_DIR)

    # ── Step 2: Feature Engineering (on raw data) ───────────────────────
    logger.info("\n▶ STEP 2: Feature Engineering")
    df_raw_train = load_dataset(DATASET_TRAIN_PATH)
    df_raw_test = load_dataset(DATASET_TEST_PATH)
    df_eng_train = feature_engineering_pipeline(df_raw_train.copy())
    df_eng_test = feature_engineering_pipeline(df_raw_test.copy())

    # ── Step 3: Preprocessing ───────────────────────────────────────────
    logger.info("\n▶ STEP 3: Data Preprocessing")
    col_info = detect_columns(df_eng_train)
    prep = preprocess_pipeline(df_eng_train, df_eng_test, col_info, OUTPUT_DIR, MODEL_DIR)

    # ── Step 4: Train Classifiers ───────────────────────────────────────
    logger.info("\n▶ STEP 4: Training Classifiers")
    training = train_all_models(
        prep["X_train"], prep["y_train"],
        prep["X_test"], prep["y_test"],
    )
    save_best_model(training["best_model"], training["best_model_name"], MODEL_DIR)

    # ── Step 4.5: Train Regressors (Salary Package LPA) ─────────────────
    logger.info("\n▶ STEP 4.5: Training Salary Package Regressors")
    reg_train = load_dataset(DATASET_TRAIN_PATH)
    reg_test = load_dataset(DATASET_TEST_PATH)
    if "Salary_LPA" in reg_train.columns and "Salary_LPA" in reg_test.columns:
        from train_regressor import train_regression_models
        from model_validation import plot_overfitting_underfitting, plot_bias_variance_tradeoff
        reg_features = [c for c in prep["feature_names"] if c in prep["X_train"].columns]
        train_regression_models(
            prep["X_train"][reg_features], reg_train["Salary_LPA"].iloc[:len(prep["X_train"])],
            prep["X_test"][reg_features], reg_test["Salary_LPA"].iloc[:len(prep["X_test"])],
            OUTPUT_DIR, MODEL_DIR
        )
        plot_overfitting_underfitting(prep["X_train"][reg_features], prep["y_train"], OUTPUT_DIR)
        plot_bias_variance_tradeoff(OUTPUT_DIR)

    # ── Step 5: Evaluate ────────────────────────────────────────────────
    logger.info("\n▶ STEP 5: Model Evaluation")
    comparison_df = evaluation_pipeline(
        training["all_models"],
        training["best_model_name"],
        prep["X_test"], prep["y_test"],
        OUTPUT_DIR,
    )

    # ── Step 6: Clustering ──────────────────────────────────────────────
    logger.info("\n▶ STEP 6: Student Clustering")
    X_all = pd.concat([prep["X_train"], prep["X_test"]], axis=0)
    cluster_result = train_cluster_pipeline(
        X_all.values,
        prep["feature_names"],
        OUTPUT_DIR,
        MODEL_DIR,
    )

    # ── Step 7: Association Rules & PCA Dimensionality Reduction ────────
    logger.info("\n▶ STEP 7: Association Rules & PCA Dimensionality Reduction")
    from association_and_pca import create_student_skill_baskets, mine_association_rules, run_pca_analysis
    baskets = create_student_skill_baskets(df_raw_train.head(5000))
    rules_res = mine_association_rules(baskets, min_support=0.15, min_confidence=0.5, output_dir=OUTPUT_DIR)
    pca_res = run_pca_analysis(prep["X_train"].values[:3000], prep["feature_names"], prep["y_train"].values[:3000], output_dir=OUTPUT_DIR)

    # ── Step 8: Reinforcement Learning Optimal Placement Strategy ───────
    logger.info("\n▶ STEP 8: Reinforcement Learning Placement Policy Convergence")
    from reinforcement_learning import train_rl_placement_agent
    rl_res = train_rl_placement_agent(episodes=600, output_dir=OUTPUT_DIR)

    # ── Step 9: Explainable AI ──────────────────────────────────────────
    logger.info("\n▶ STEP 9: Explainable AI")
    xai = explanation_pipeline(
        training["best_model"],
        prep["X_test"],
        prep["feature_names"],
        OUTPUT_DIR,
    )

    # ── Save feature names ──────────────────────────────────────────────
    joblib.dump(prep["feature_names"], os.path.join(MODEL_DIR, "feature_names.joblib"))
    logger.info("Feature names saved to models/feature_names.joblib")

    logger.info("\n" + "=" * 70)
    logger.info("   TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)

    return {
        "prep": prep,
        "training": training,
        "cluster_result": cluster_result,
        "xai": xai,
        "comparison_df": comparison_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Single Student Prediction
# ═══════════════════════════════════════════════════════════════════════════
def predict_single_student(
    student_data: dict[str, float],
    best_model=None,
    scaler=None,
    encoders=None,
    cluster_model=None,
    cluster_names=None,
    feature_names=None,
    resume_text: str = "",
    interview_scores: dict | None = None,
) -> str:
    """Run a full prediction for a single student.

    Args:
        student_data: Raw feature values (unscaled).
        best_model: Fitted classifier (or loads from MODEL_DIR).
        scaler: Fitted scaler (or loads from MODEL_DIR).
        encoders: Fitted encoders (or loads from MODEL_DIR).
        cluster_model: Fitted KMeans (or loads from MODEL_DIR).
        cluster_names: Cluster name mapping (or loads from MODEL_DIR).
        feature_names: Feature column names (or loads from MODEL_DIR).
        resume_text: Optional resume text for ATS analysis.
        interview_scores: Optional dict with technical, communication,
            confidence, question_accuracy scores.

    Returns:
        Full report string.
    """
    # Load models if not provided
    if best_model is None:
        best_model = joblib.load(os.path.join(MODEL_DIR, "best_classifier.joblib"))
    if scaler is None:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    if encoders is None:
        encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    if cluster_model is None:
        cluster_model = joblib.load(os.path.join(MODEL_DIR, "cluster_model.joblib"))
    if cluster_names is None:
        cluster_names = joblib.load(os.path.join(MODEL_DIR, "cluster_names.joblib"))
    if feature_names is None:
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

    # ── Build feature vector ────────────────────────────────────────────
    # Apply feature engineering to create composite scores
    eng_data = dict(student_data)

    # Create composite features inline
    for feat_name, spec in COMPOSITE_FEATURES.items():
        available = [(c, w) for c, w in zip(spec["columns"], spec["weights"]) if c in eng_data]
        if available:
            cols, weights = zip(*available)
            vals = [eng_data[c] for c in cols]
            # Simple normalized weighted sum
            max_vals = {"CGPA": 10, "Backlogs": 5, "Coding_Skills": 10,
                        "Aptitude_Test_Score": 100, "Projects": 10,
                        "Internships": 5, "Certifications": 10, "Age": 30,
                        "Communication_Skills": 10, "Soft_Skills_Rating": 10}
            norm_vals = [v / max_vals.get(c, max(v, 1)) for c, v in zip(cols, vals)]
            score = sum(w * nv for w, nv in zip(weights, norm_vals))
            eng_data[feat_name] = max(0, min(1, score))

    # Overall skill score
    overall_feats = ["AcademicScore", "TechnicalScore", "ExperienceScore",
                     "CommunicationScore", "ProjectScore", "CodingIndex"]
    overall_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
    available_overall = [(f, w) for f, w in zip(overall_feats, overall_weights) if f in eng_data]
    if available_overall:
        feats, wts = zip(*available_overall)
        total_w = sum(wts)
        eng_data["OverallSkillScore"] = sum(eng_data[f] * w / total_w for f, w in zip(feats, wts))

    # Encode categoricals
    for col, le in encoders.items():
        if col in eng_data:
            val = str(eng_data[col])
            if val in le.classes_:
                eng_data[col] = le.transform([val])[0]
            else:
                eng_data[col] = 0  # Unknown category fallback

    # Build feature DataFrame in the right order for the model
    feature_df = pd.DataFrame([{f: eng_data.get(f, 0.0) for f in feature_names}])

    # Scale using the exact columns and order the scaler was fitted on
    if hasattr(scaler, "feature_names_in_"):
        scale_cols = scaler.feature_names_in_
        # This replaces the values in-place while keeping feature_df's column order intact
        feature_df[scale_cols] = scaler.transform(feature_df[scale_cols])
        feature_scaled = feature_df
    else:
        feature_scaled = scaler.transform(feature_df)

    # ── Predict ─────────────────────────────────────────────────────────
    probability = float(best_model.predict_proba(feature_scaled)[0][1])
    readiness = classify_readiness(probability)

    # ── Cluster ─────────────────────────────────────────────────────────
    cluster_info = predict_cluster(feature_scaled, cluster_model, cluster_names)

    # ── Explanation ─────────────────────────────────────────────────────
    explanation = explain_single_prediction(
        best_model, feature_scaled, feature_names
    )

    # ── Skill Gap ───────────────────────────────────────────────────────
    # Use normalized values for skill analysis
    norm_data = {}
    max_vals = {"CGPA": 10, "Backlogs": 5, "Coding_Skills": 10,
                "Aptitude_Test_Score": 100, "Projects": 10,
                "Internships": 5, "Certifications": 10, "Age": 30,
                "Communication_Skills": 10, "Soft_Skills_Rating": 10}
    for k, v in student_data.items():
        norm_data[k] = v / max_vals.get(k, max(v, 1)) if isinstance(v, (int, float)) else 0
    for k, v in eng_data.items():
        if k not in norm_data and isinstance(v, (int, float)):
            norm_data[k] = v

    skill_analysis = analyse_skill_gaps(norm_data)
    roadmap = generate_roadmap(skill_analysis["weaknesses"], skill_analysis["strengths"])

    # ── Company Matching ────────────────────────────────────────────────
    company_result = match_companies(probability, norm_data)

    # ── Resume Analysis (optional) ──────────────────────────────────────
    resume_result = score_resume(resume_text) if resume_text.strip() else None

    # ── Interview Evaluation (optional) ─────────────────────────────────
    interview_result = None
    if interview_scores:
        interview_result = evaluate_interview(**interview_scores)

    # ── Salary Package Regression (LPA) ─────────────────────────────────
    predicted_salary = None
    reg_path = os.path.join(MODEL_DIR, "best_regressor.joblib")
    if os.path.isfile(reg_path):
        try:
            reg_model = joblib.load(reg_path)
            if hasattr(reg_model, "feature_names_in_"):
                reg_cols = [c for c in reg_model.feature_names_in_ if c in feature_scaled.columns]
                reg_input = feature_scaled[reg_cols]
            else:
                reg_input = feature_scaled
            raw_salary = float(reg_model.predict(reg_input)[0])
            predicted_salary = round(max(0.0, raw_salary), 2)
        except Exception as e:
            logger.warning("Could not predict salary: %s", e)

    # ── Generate Report ─────────────────────────────────────────────────
    report = generate_report(
        probability=probability,
        cluster_name=cluster_info["cluster_name"],
        skill_analysis=skill_analysis,
        roadmap=roadmap,
        company_matching=company_result,
        explanation=explanation,
        resume_analysis=resume_result,
        interview_result=interview_result,
        predicted_salary=predicted_salary,
    )

    # Save report
    save_report(report, OUTPUT_DIR)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Example Prediction
# ═══════════════════════════════════════════════════════════════════════════
def run_example_prediction(pipeline_data: dict | None = None) -> None:
    """Run a sample prediction to demonstrate the system.

    Args:
        pipeline_data: Training pipeline output (optional).
    """
    logger.info("\n" + "=" * 70)
    logger.info("   EXAMPLE PREDICTION")
    logger.info("=" * 70)

    # Sample student profile
    student = {
        "Age": 22,
        "Gender": "Male",
        "Degree": "B.Tech",
        "Branch": "CSE",
        "CGPA": 7.8,
        "Internships": 1,
        "Projects": 4,
        "Coding_Skills": 7,
        "Communication_Skills": 8,
        "Aptitude_Test_Score": 75,
        "Soft_Skills_Rating": 8,
        "Certifications": 2,
        "Backlogs": 0,
    }

    # Sample resume
    resume = """
    Rahul Sharma
    Email: rahul@email.com | GitHub: github.com/rahul
    LinkedIn: linkedin.com/in/rahul

    Summary
    B.Tech CSE student passionate about software development and machine learning.

    Education
    B.Tech Computer Science — ABC University | CGPA: 7.8

    Experience
    Software Intern — XYZ Corp (Summer 2024)
    - Developed REST APIs using Python Flask
    - Implemented CI/CD pipelines using Docker

    Projects
    - Built a placement prediction ML model using scikit-learn
    - Developed a chat application using React and Node.js
    - Created an automated web scraper using Python

    Skills
    Python, Java, SQL, Machine Learning, Docker, Git, Linux

    Certifications
    - Google Cloud Fundamentals
    """

    # Sample interview scores
    interview = {
        "technical_score": 72,
        "communication_score": 58,
        "confidence_score": 65,
        "question_accuracy": 70,
    }

    kwargs = {}
    if pipeline_data:
        kwargs = {
            "best_model": pipeline_data["training"]["best_model"],
            "scaler": pipeline_data["prep"]["scaler"],
            "encoders": pipeline_data["prep"]["encoders"],
            "cluster_model": pipeline_data["cluster_result"]["model"],
            "cluster_names": pipeline_data["cluster_result"]["cluster_names"],
            "feature_names": pipeline_data["prep"]["feature_names"],
        }

    report = predict_single_student(
        student_data=student,
        resume_text=resume,
        interview_scores=interview,
        **kwargs,
    )

    print(report)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Interactive Prediction
# ═══════════════════════════════════════════════════════════════════════════
def run_interactive_prediction() -> None:
    """Prompt user for student details and generate a report."""
    logger.info("\n" + "=" * 70)
    logger.info("   INTERACTIVE PREDICTION MODE")
    logger.info("=" * 70)
    print("\nPlease enter the student's details:")

    try:
        student = {
            "Age": int(input("Age (e.g., 22): ") or 22),
            "Gender": input("Gender (Male/Female): ") or "Male",
            "Degree": input("Degree (e.g., B.Tech): ") or "B.Tech",
            "Branch": input("Branch (e.g., CSE): ") or "CSE",
            "CGPA": float(input("CGPA (0-10): ") or 7.5),
            "Internships": int(input("Number of Internships: ") or 0),
            "Projects": int(input("Number of Projects: ") or 0),
            "Coding_Skills": int(input("Coding Skills (0-10): ") or 5),
            "Communication_Skills": int(input("Communication Skills (0-10): ") or 5),
            "Aptitude_Test_Score": int(input("Aptitude Score (0-100): ") or 60),
            "Soft_Skills_Rating": int(input("Soft Skills Rating (0-10): ") or 5),
            "Certifications": int(input("Number of Certifications: ") or 0),
            "Backlogs": int(input("Current Backlogs: ") or 0),
        }
    except ValueError:
        logger.error("Invalid input. Please enter numbers where expected.")
        sys.exit(1)

    print("\nGenerating report...\n")

    # Try LLM-based resume classification if model exists
    llm_result = None
    llm_model_path = os.path.join(MODEL_DIR, "llm_resume_model")
    if LLM_AVAILABLE and os.path.isdir(llm_model_path):
        try:
            from llm_resume_classifier import predict_resume_llm
            resume_paragraph = llm_generate_text(pd.Series(student))
            llm_result = predict_resume_llm(resume_paragraph, model_path=llm_model_path)
            logger.info(f"LLM Prediction: {llm_result['label']} (confidence: {llm_result['confidence']:.2%})")
        except Exception as e:
            logger.warning(f"LLM prediction failed (non-critical): {e}")

    report = predict_single_student(student_data=student)

    # Append LLM result to the report if available
    if llm_result:
        llm_section = (
            "\n────────────────────────────────────────────────────────────\n"
            "\n  \U0001f9e0 LLM RESUME ANALYSIS (DistilBERT)\n"
            "  ────────────────────────────────────────\n"
            f"  LLM Prediction   : {llm_result['label']}\n"
            f"  LLM Confidence   : {llm_result['confidence']:.1%}\n"
            f"  Placed Probability: {llm_result['placed_probability']:.1%}\n"
        )
        # Insert before the disclaimer
        report = report.replace(
            "  \u2139\ufe0f  This report is generated",
            llm_section + "\n  \u2139\ufe0f  This report is generated"
        )

    print(report)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════
def run_batch_prediction(input_csv: str) -> None:
    """Run batch prediction on a CSV file and output results."""
    if not os.path.isfile(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    logger.info(f"Loading batch data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Load models
    best_model = joblib.load(os.path.join(MODEL_DIR, "best_classifier.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    cluster_model = joblib.load(os.path.join(MODEL_DIR, "cluster_model.joblib"))
    cluster_names = joblib.load(os.path.join(MODEL_DIR, "cluster_names.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

    logger.info("Running predictions...")
    results = []
    
    # Simple loop for demonstration; vectorized is better for huge datasets
    for _, row in df.iterrows():
        # Get report string, but we just want to extract basic info
        # We can run a simplified version of the logic
        student_data = row.to_dict()
        eng_data = dict(student_data)

        # Feature Engineering
        for feat_name, spec in COMPOSITE_FEATURES.items():
            available = [(c, w) for c, w in zip(spec["columns"], spec["weights"]) if c in eng_data]
            if available:
                cols, weights = zip(*available)
                vals = [eng_data[c] for c in cols]
                max_vals = {"CGPA": 10, "Backlogs": 5, "Coding_Skills": 10,
                            "Aptitude_Test_Score": 100, "Projects": 10,
                            "Internships": 5, "Certifications": 10, "Age": 30,
                            "Communication_Skills": 10, "Soft_Skills_Rating": 10}
                norm_vals = [v / max_vals.get(c, max(v, 1)) for c, v in zip(cols, vals)]
                score = sum(w * nv for w, nv in zip(weights, norm_vals))
                eng_data[feat_name] = max(0, min(1, score))

        overall_feats = ["AcademicScore", "TechnicalScore", "ExperienceScore", "CommunicationScore", "ProjectScore", "CodingIndex"]
        overall_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        available_overall = [(f, w) for f, w in zip(overall_feats, overall_weights) if f in eng_data]
        if available_overall:
            feats, wts = zip(*available_overall)
            total_w = sum(wts)
            eng_data["OverallSkillScore"] = sum(eng_data[f] * w / total_w for f, w in zip(feats, wts))

        for col, le in encoders.items():
            if col in eng_data:
                val = str(eng_data[col])
                eng_data[col] = le.transform([val])[0] if val in le.classes_ else 0

        feature_df = pd.DataFrame([{f: eng_data.get(f, 0.0) for f in feature_names}])
        
        if hasattr(scaler, "feature_names_in_"):
            scale_cols = scaler.feature_names_in_
            feature_df[scale_cols] = scaler.transform(feature_df[scale_cols])
            feature_scaled = feature_df
        else:
            feature_scaled = scaler.transform(feature_df)
        
        prob = float(best_model.predict_proba(feature_scaled)[0][1])
        readiness = classify_readiness(prob)
        cluster_info = predict_cluster(feature_scaled, cluster_model, cluster_names)
        
        results.append({
            "Probability": round(prob * 100, 2),
            "Readiness_Level": readiness,
            "Cluster": cluster_info["cluster_name"]
        })

    # Combine back to df
    out_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    out_path = os.path.join(OUTPUT_DIR, "batch_predictions.csv")
    out_df.to_csv(out_path, index=False)
    logger.info(f"Batch prediction complete! Saved to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Main entry point."""
    # Fix Windows console encoding for Unicode output (box-drawing, emojis)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="PlacementIQ AI Pipeline")
    parser.add_argument("--train", action="store_true",
                        help="Run the full training pipeline")
    parser.add_argument("--train-llm", action="store_true",
                        help="Fine-tune the DistilBERT LLM model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive mode to predict for a single student")
    parser.add_argument("--batch", type=str, metavar="CSV_FILE",
                        help="Run batch prediction on a provided CSV file")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_prediction()
    elif args.batch:
        run_batch_prediction(args.batch)
    elif args.train_llm:
        try:
            from llm_resume_classifier import train_llm_model
        except ImportError:
            logger.error("PyTorch and Transformers are required for LLM training.")
            logger.error("Install them with: pip install torch transformers")
            sys.exit(1)
        ensure_datasets()
        train_llm_model(DATASET_TRAIN_PATH, MODEL_DIR)
    elif args.train:
        pipeline_data = run_training_pipeline()
        logger.info("\n✅ Model training complete!")
        logger.info("   Outputs: %s", OUTPUT_DIR)
        logger.info("   Models:  %s", MODEL_DIR)
    else:
        logger.info("No action specified. Use --interactive, --batch <file>, --train, or --train-llm.")
        parser.print_help()


if __name__ == "__main__":
    main()
