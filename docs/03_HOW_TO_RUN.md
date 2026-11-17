# 🚀 How to Run PlacementIQ AI — Complete Execution Guide
**Continuous Evaluation Test 2 (CE-2) | Course Code: 24CSE0316**  
**Course Title:** Artificial Intelligence & Machine Learning  

This guide provides step-by-step instructions to set up the environment, run the training pipeline, test every individual module, execute the 7 phase Jupyter notebooks, and verify with automated test suites.

---

## ⚡ The 3-Minute Quickstart

If you need to quickly test or demonstrate the project:

```powershell
# 1. Navigate to project root
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI

# 2. Run Complete Verification Test Suite (Tests all 7 syllabus phases in ~30s)
python test_full_syllabus_suite.py

# 3. Run Interactive Student Prediction (Dual target: Placement % + Salary Package in LPA)
python src/predict.py --interactive

# 4. Run Full Pipeline Training
python src/predict.py --train
```

---

## 📋 System Prerequisites

| Requirement | Supported Version | How to Verify |
|---|---|---|
| **Python** | 3.10 – 3.15 | `python --version` |
| **pip** | Latest | `pip --version` |
| **Core Libraries** | `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `mlxtend`, `nbformat` | `pip list` |

---

## 🛠️ Environment Setup

### Activate Virtual Environment
```powershell
# From project directory:
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI

# If using PowerShell and script execution is restricted:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate venv:
.\venv\Scripts\Activate.ps1

# (Or in CMD: venv\Scripts\activate.bat | Git Bash: source venv/Scripts/activate)

# Install / update all dependencies:
pip install -r requirements.txt
```

*(Note: If using Python directly without activating venv, use `C:\Python315\python.exe`)*

---

## 💻 Running the Central CLI Pipeline (`src/predict.py`)

`predict.py` orchestrates data loading, training, evaluation, clustering, recommendations, and prediction.

### 1. Interactive Single-Student Prediction
Prompts for student details and outputs both **Placement Probability** and **Estimated Salary Package (LPA)**:
```powershell
python src/predict.py --interactive
```

### 2. Full Training Pipeline
Retrains all regression models, 6 classifiers, clustering, association rules, PCA, and reinforcement learning:
```powershell
python src/predict.py --train
```

### 3. Batch Predictions on CSV
Generates predictions for a CSV file of students and outputs `outputs/batch_predictions.csv`:
```powershell
python src/predict.py --batch dataset/test.csv
```

---

## 📓 Running the 7 Phase Jupyter Notebooks

Located directly in the root of `PlacementIQ_AI`:

```text
├── 01_Phase1_Data_Preprocessing_and_EDA.ipynb
├── 02_Phase2_Supervised_Regression.ipynb
├── 03_Phase3_Supervised_Classification.ipynb
├── 04_Phase4_Model_Evaluation_and_Validation.ipynb
├── 05_Phase5_Unsupervised_Clustering.ipynb
├── 06_Phase6_Association_Rules_and_Recommendations.ipynb
└── 07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb
```

### How to Run:
- **VS Code:** Open `PlacementIQ_AI`, open any notebook, select Python Kernel, and click **"Run All"**.
- **Jupyter Notebook / Lab:**
  ```powershell
  jupyter notebook
  ```
- **Google Colab:** Upload the notebook and `dataset/train.csv` + `dataset/test.csv`, install dependencies, and click **"Run all"**.

---

## 🔬 Running Individual Python Modules Directly

Every topic module can be run as a standalone script:

```powershell
# Phase 2: Supervised Learning — Regression (Linear, Multiple, Polynomial, MAE, MSE, RMSE, R²)
python src/train_regressor.py

# Phase 3: Supervised Learning — Classification (Logistic, KNN, DecisionTree, RandomForest, NaiveBayes, SVM)
python src/train_classifier.py

# Phase 4: Model Evaluation & Validation (5-Fold Stratified CV, Overfitting Curves, Confusion Matrix)
python src/model_validation.py

# Phase 5: Unsupervised Learning — Clustering (K-Means, Ward Linkage Dendrogram, DBSCAN)
python src/unsupervised_learning.py

# Phase 6: Association Rule Mining & PCA (Apriori, FP-Growth, Recommender, Scree Plot, 2D Scatter)
python src/association_and_pca.py

# Phase 7: Reinforcement Learning (Student MDP, Q-Learning Agent, Optimal Policy, AI Ethics)
python src/reinforcement_learning.py
```

---

## 🧪 Running Automated Test Suites

```powershell
# 1. Run Complete Syllabus Suite (All 7 Phases)
python test_full_syllabus_suite.py

# 2. Run Comprehensive Curriculum Deep Test
python test_all_curriculum_modules.py
```

---

## 📁 Output Artifacts Generated

- **Saved Models (`models/`):**
  - `best_classifier.joblib`: Random Forest Classifier
  - `best_regressor.joblib`: Multiple Linear Regressor (LPA Package)
  - `cluster_model.joblib`: K-Means Cluster Model
  - `scaler.joblib` & `encoders.joblib`: Data transformers
- **Visualizations & Plots (`outputs/`):**
  - `regression_actual_vs_predicted.png` & `regression_residuals.png`
  - `overfitting_underfitting_curve.png` & `bias_variance_tradeoff.png`
  - `detailed_confusion_matrix.png`
  - `hierarchical_dendrogram.png` & `cluster_elbow_silhouette.png`
  - `pca_variance_and_projection.png`
  - `rl_qlearning_convergence.png`
  - `model_comparison.png` & `roc_curves_comparison.png`
  - `correlation_heatmap.png`
