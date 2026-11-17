# 🚀 Master Execution Guide: Running Every Module, Script & Notebook
**PlacementIQ AI — Continuous Evaluation (CE-2) Project Platform**  
**Course Code:** 24CSE0316 | **Course Title:** Artificial Intelligence & Machine Learning  

---

## ⚡ Quick Cheat Sheet: The 3 Most Important Commands

If you need to quickly demonstrate the working project to your professor or evaluator:

```powershell
# 1. Run Complete Automated Test Suite (Verifies all 7 Syllabus Phases in ~30 seconds)
python test_full_syllabus_suite.py

# 2. Run Full End-to-End Training Pipeline (Classification, Regression, Clustering, PCA, RL)
python src/predict.py --train

# 3. Run Interactive Prediction (Prompts student details and outputs Probability + Package in LPA)
python src/predict.py --interactive
```

---

## 📑 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Running the 7 Phase Jupyter Notebooks](#2-running-the-7-phase-jupyter-notebooks)
3. [Running the Full Pipeline CLI (`src/predict.py`)](#3-running-the-full-pipeline-cli-srcpredictpy)
4. [Running Individual Python Modules Directly](#4-running-individual-python-modules-directly)
5. [Running the Automated Verification Test Suites](#5-running-the-automated-verification-test-suites)
6. [Outputs and Model Files Generated](#6-outputs-and-model-files-generated)
7. [Running on Google Colab](#7-running-on-google-colab)
8. [Troubleshooting & FAQs](#8-troubleshooting--faqs)

---

## 1. Prerequisites & Environment Setup

### Check Python Installation
Make sure you have Python 3.10, 3.11, 3.12, 3.13, 3.14, or 3.15 installed:
```powershell
python --version
```

### Option A: Using Project Virtual Environment
```powershell
# Navigate to project root
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI

# Create virtual environment (if not already created)
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run this first:)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate on Command Prompt (cmd)
venv\Scripts\activate.bat

# Activate on Git Bash
source venv/Scripts/activate

# Install all required dependencies
pip install -r requirements.txt
```

### Option B: Using System Python (e.g. Python 3.15)
If your venv is inactive, you can invoke your installed Python directly:
```powershell
C:\Python315\python.exe -m pip install -r requirements.txt
```

---

## 2. Running the 7 Phase Jupyter Notebooks

Every major syllabus topic has its own dedicated, documented, and fully executable Jupyter Notebook located in the project root:

```text
PlacementIQ_AI/
├── 01_Phase1_Data_Preprocessing_and_EDA.ipynb
├── 02_Phase2_Supervised_Regression.ipynb
├── 03_Phase3_Supervised_Classification.ipynb
├── 04_Phase4_Model_Evaluation_and_Validation.ipynb
├── 05_Phase5_Unsupervised_Clustering.ipynb
├── 06_Phase6_Association_Rules_and_Recommendations.ipynb
└── 07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb
```

### How to Launch the Notebooks

#### In VS Code (Recommended):
1. Open the `PlacementIQ_AI` folder in VS Code.
2. Click on any notebook (e.g., `02_Phase2_Supervised_Regression.ipynb`).
3. Select your Python Kernel in the top right corner (`Python 3.15` or `venv`).
4. Click **"Run All"** or execute cells individually with `Shift + Enter`.

#### In Classic Jupyter Notebook / JupyterLab:
```powershell
# Launch Jupyter server
jupyter notebook
```
A browser tab will open at `http://localhost:8888`. Click on any phase notebook to run it.

---

### Phase-by-Phase Notebook Summary

| Notebook File | Main Syllabus Topics Covered | What It Demonstrates |
| :--- | :--- | :--- |
| **`01_Phase1_Data_Preprocessing_and_EDA.ipynb`** | EDA, Missing Values, Outliers, 80/20 Split | Data distribution plots, IQR boxplots, correlation heatmaps, `StandardScaler`. |
| **`02_Phase2_Supervised_Regression.ipynb`** | Supervised Regression: Linear, Multiple, Polynomial | Predicts `Salary_LPA` package; computes **MAE (0.36 LPA)**, **MSE**, **RMSE**, and **$R^2$ (0.9912)** with residual plots. |
| **`03_Phase3_Supervised_Classification.ipynb`** | Classification: Logistic, KNN, Tree, Forest, NB, SVM | Benchmarks all 6 classifiers predicting `Placement_Status` with accuracy and ROC comparison bar charts. |
| **`04_Phase4_Model_Evaluation_and_Validation.ipynb`** | Overfitting/Underfitting, Bias-Variance, Confusion Matrix | Validation curves across tree depths (1–15), 5-Fold Stratified CV, error decomposition, and annotated confusion matrix. |
| **`05_Phase5_Unsupervised_Clustering.ipynb`** | Clustering: K-Means, Hierarchical, DBSCAN | WCSS Elbow plot, Silhouette curves, Ward Linkage Dendrogram, DBSCAN density noise filtering, student cohort segmentation. |
| **`06_Phase6_Association_Rules_and_Recommendations.ipynb`**| Association Rules, FP-Growth, Recommendations, PCA | Mined 149 frequent itemsets, placement rules ($\text{Lift} > 1.8$), content-based course recommender, PCA Scree plot & 2D projection. |
| **`07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb`**| Reinforcement Learning (RL), MDP, Q-Learning | Student preparation MDP environment, Q-table Bellman convergence curve, optimal policy table, Comparative Study matrix, and AI Ethics. |

---

## 3. Running the Full Pipeline CLI (`src/predict.py`)

`predict.py` is the central command-line orchestrator connecting all modules:

### 1. Run Full Training Pipeline
Executes preprocessing, feature engineering, classification training, salary regression, clustering, association rules, PCA, and RL:
```powershell
python src/predict.py --train
```
**Expected Output:**
- Prints training logs for all models.
- Generates and saves all `.joblib` estimators in `models/`.
- Generates all visualization PNGs and CSV reports in `outputs/`.

---

### 2. Run Interactive Student Prediction (Single Student Mode)
Prompts for student details and outputs both **Placement Probability** and **Estimated Salary Package in LPA**:
```powershell
python src/predict.py --interactive
```

**Example Interactive Session:**
```text
Age (e.g., 22): 22
Gender (Male/Female): Male
Degree (e.g., B.Tech): B.Tech
Branch (e.g., CSE): CSE
CGPA (0-10): 8.5
Number of Internships: 2
Number of Projects: 4
Coding Skills (0-10): 8
Communication Skills (0-10): 8
Aptitude Score (0-100): 85
Soft Skills Rating (0-10): 8
Number of Certifications: 3
Current Backlogs: 0
```

**Generated Report Snippet:**
```text
╔══════════════════════════════════════════════════════════╗
║       PLACEMENTIQ AI — PLACEMENT READINESS REPORT        ║
╚══════════════════════════════════════════════════════════╝

  📅 Generated: 2026-11-16 10:30:00
  👤 Student:   Student

────────────────────────────────────────────────────────────

  📊 PLACEMENT PREDICTION & PACKAGE ESTIMATION
  ────────────────────────────────────────
  Placement Probability : 75.1%
  Readiness Level       : Placement Ready
  Student Cluster       : Top Performers
  Estimated Package/CTC : 26.95 LPA

────────────────────────────────────────────────────────────

  🔍 PREDICTION EXPLANATION
  ────────────────────────────────────────
  Factors increasing placement chance:
    ✔ Communication_Skills (contribution: +0.3836)
    ✔ CommunicationScore (contribution: +0.1680)
    ✔ CGPA (contribution: +0.0806)
    ✔ Coding_Skills (contribution: +0.0639)

  🎯 4-WEEK ROADMAP:
    • Week 1: Coding & Technical Skills (DSA, LeetCode)
    • Week 2: Projects & Portfolio (Full-Stack Deployment)
    • Week 3: Communication & Soft Skills (GD, Presentation)
    • Week 4: Interview Preparation (Mock HR & Behavioral)
```

---

### 3. Run Batch Predictions on a CSV File
Predicts placement probability and readiness level for thousands of students at once:
```powershell
python src/predict.py --batch dataset/test.csv
```
Outputs `outputs/batch_predictions.csv` with added `Probability`, `Readiness_Level`, and `Cluster` columns.

---

### 4. Fine-Tune DistilBERT LLM on Resume Text (Optional Deep Learning Module)
```powershell
python src/predict.py --train-llm
```

---

## 4. Running Individual Python Modules Directly

Each file in `src/` can be run as an independent, standalone script:

```powershell
# 1. Test Supervised Regression (Simple, Multiple, Polynomial, MAE, MSE, RMSE, R²)
python src/train_regressor.py

# 2. Test Supervised Classification (Logistic, KNN, Tree, Forest, Naive Bayes, SVM)
python src/train_classifier.py

# 3. Test Model Validation & Curves (Overfitting/Underfitting & Bias-Variance)
python src/model_validation.py

# 4. Test Unsupervised Clustering (K-Means, Dendrogram, DBSCAN)
python src/unsupervised_learning.py

# 5. Test Association Rule Mining & PCA (Apriori, FP-Growth, Scree Plot, 2D Scatter)
python src/association_and_pca.py

# 6. Test Reinforcement Learning (Placement MDP, Q-Learning Agent & Optimal Policy)
python src/reinforcement_learning.py

# 7. Test Feature Engineering (Composite Score Construction)
python src/feature_engineering.py

# 8. Test Data Preprocessing Pipeline
python src/preprocessing.py

# 9. Test Rule-Based Resume ATS Scoring
python src/resume_scoring.py

# 10. Test Mock Interview Scoring
python src/interview_scoring.py

# 11. Test Skill Gap Analysis & 4-Week Roadmap Generator
python src/recommendation_engine.py
```

---

## 5. Running the Automated Verification Test Suites

We have provided automated test runners that run all components and assert that metrics pass syllabus thresholds:

### Master Syllabus Suite (All 7 Phases)
```powershell
python test_full_syllabus_suite.py
```
**Output:**
```text
================================================================================
🎓 RUNNING COMPLETE AIML CONTINUOUS EVALUATION (CE-2) TEST SUITE
================================================================================
▶ Phase 1: Checking Data Preprocessing & 80/20 Stratified Partitioning...
  ✔ Train set: 40,000 rows (80%), Test set: 10,000 rows (20%) — PASS
▶ Phase 2: Supervised Learning — Regression (Linear, Multiple, Polynomial, MAE, MSE, RMSE, R²)...
  ✔ Regression models trained successfully! Best R² Score: 0.9912 — PASS
▶ Phase 3: Supervised Learning — Classification (Logistic, KNN, Tree, Forest, Naive Bayes, SVM)...
  ✔ All 6 syllabus classification models present in registry — PASS
▶ Phase 4: Model Evaluation & Validation (Stratified CV, Overfitting Curve, Confusion Matrix)...
  ✔ Cross-Validation & Overfitting/Underfitting diagnostics generated — PASS
▶ Phase 5: Unsupervised Learning — Clustering (K-Means, Hierarchical, DBSCAN, Silhouette)...
  ✔ K-Means, Hierarchical & DBSCAN clustering evaluated — PASS
▶ Phase 6: Association Rules (Apriori/FP-Growth), Recommendations & PCA...
  ✔ FP-Growth rules & PCA Scree / 2D projection generated — PASS
▶ Phase 7: Reinforcement Learning (MDP, Q-Learning Agent, Policy, Ethics)...
  ✔ Q-Learning agent converged to optimal placement policy — PASS
================================================================================
🎉 ALL 7 SYLLABUS PHASES VERIFIED AND PASSED WITH 100% SUCCESS!
================================================================================
```

---

## 6. Outputs and Model Files Generated

### Visualizations & Plots (`outputs/`)
All plots are rendered at high resolution ($150$ DPI) in non-interactive headless mode:
- `regression_actual_vs_predicted.png`: Regression actual vs. predicted salary scatter plot.
- `regression_residuals.png`: Distribution of residual prediction errors.
- `overfitting_underfitting_curve.png`: Decision tree depth complexity curve.
- `bias_variance_tradeoff.png`: Theoretical decomposition curve.
- `detailed_confusion_matrix.png`: Heatmap with annotated TP, TN, FP, FN.
- `hierarchical_dendrogram.png`: Hierarchical tree with Ward linkage.
- `pca_variance_and_projection.png`: Scree plot and 2D student feature space projection.
- `rl_qlearning_convergence.png`: Q-Learning rolling cumulative reward convergence.
- `roc_curves_comparison.png`: Comparison of ROC curves across classifiers.
- `model_comparison.png`: Grouped bar chart comparing classification metrics.
- `cluster_elbow_silhouette.png`: Optimal K selection curves.
- `correlation_heatmap.png`: Full correlation matrix heatmap.
- `histograms.png` & `boxplots.png`: Distribution and outlier visualizations.

### Saved Models & Transformers (`models/`)
- `best_classifier.joblib`: Top-performing classification model (Random Forest).
- `best_regressor.joblib`: Top-performing salary regression model (Multiple Linear Regression).
- `regressor_meta.joblib`: Metadata containing regression evaluation metrics (MAE, RMSE, $R^2$).
- `cluster_model.joblib`: Fitted K-Means cluster estimator.
- `cluster_names.joblib`: Mapping of cluster IDs to student segment names.
- `scaler.joblib`: Fitted `StandardScaler` transformer.
- `encoders.joblib`: Fitted `LabelEncoder` transformers.
- `feature_names.joblib`: Exact feature ordering list.

---

## 7. Running on Google Colab

If you want to demonstrate the project on Google Colab:
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/)
2. Upload any of the 7 phase notebooks (e.g., `02_Phase2_Supervised_Regression.ipynb`).
3. Upload `dataset/train.csv` and `dataset/test.csv` to Colab's session storage.
4. Run:
   ```python
   !pip install mlxtend scikit-learn pandas numpy matplotlib seaborn
   ```
5. Click **"Run all"** to execute the complete pipeline in the cloud.

---

## 8. Troubleshooting & FAQs

### Q: "Activate.ps1 cannot be loaded because running scripts is disabled on this system"
**Fix:** Run PowerShell as Administrator (or in current session) and allow script execution:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q: "ModuleNotFoundError: No module named 'mlxtend'"
**Fix:** Install the missing library via:
```powershell
pip install mlxtend
# or using specific Python:
C:\Python315\python.exe -m pip install mlxtend
```

### Q: "Matplotlib UserWarning: Figure includes Axes that are not compatible"
**Explanation:** This is a cosmetic warning from Matplotlib when rendering subplots. All charts save completely into the `outputs/` folder.

### Q: Where do I find the viva questions and theory?
**Answer:** Read [`docs/14_CE_TEST_2_EVALUATION_GUIDE_AND_VIVA.md`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/docs/14_CE_TEST_2_EVALUATION_GUIDE_AND_VIVA.md) for full answers to common viva questions on Bias-Variance, MAE/MSE/RMSE/R², Stratified CV, DBSCAN, FP-Growth, and Q-Learning.

