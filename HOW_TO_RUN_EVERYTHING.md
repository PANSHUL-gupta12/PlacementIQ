# 📖 COMPLETE MASTER GUIDE: RUNNING EVERYTHING IN PLACEMENTIQ AI
**Continuous Evaluation Test 2 (CE-2) | Course Code: 24CSE0316**  
**Course Title:** Artificial Intelligence & Machine Learning  
**Dataset Architecture:** 80/20 Stratified Partition (40,000 Train / 10,000 Test)  
**Dual Targets:** Classification (`Placement_Status`) & Regression (`Salary_LPA`)

---

## ⚡ The 3-Minute Quickstart (Top 3 Commands)

If you are standing before your professor or evaluator and need to demonstrate the entire project immediately:

```powershell
# Step 0: Always navigate into the project directory first!
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI

# Command 1: Run the Complete Automated Syllabus Verification Suite (Verifies all 7 phases in ~30 sec)
python test_full_syllabus_suite.py

# Command 2: Run Interactive Student Prediction (Prompts for marks and predicts Placement % + Salary Package LPA)
python src/predict.py --interactive

# Command 3: Run Full End-to-End Retraining Pipeline (Trains Regression, 6 Classifiers, Clustering, PCA, RL)
python src/predict.py --train
```

---

## 📁 Project Layout & What Can Be Run

```text
PlacementIQ_AI/
│
├── 📓 7 STANDALONE PHASE JUPYTER NOTEBOOKS:
│   ├── 01_Phase1_Data_Preprocessing_and_EDA.ipynb
│   ├── 02_Phase2_Supervised_Regression.ipynb
│   ├── 03_Phase3_Supervised_Classification.ipynb
│   ├── 04_Phase4_Model_Evaluation_and_Validation.ipynb
│   ├── 05_Phase5_Unsupervised_Clustering.ipynb
│   ├── 06_Phase6_Association_Rules_and_Recommendations.ipynb
│   └── 07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb
│
├── 🐍 STANDALONE EXECUTABLE PYTHON MODULES (src/):
│   ├── src/predict.py                  # Master CLI Orchestrator (--train, --interactive, --batch)
│   ├── src/train_regressor.py          # Linear, Multiple, Polynomial Regression + MAE/MSE/RMSE/R²
│   ├── src/train_classifier.py         # Logistic, KNN, DecisionTree, RandomForest, NaiveBayes, SVM
│   ├── src/model_validation.py         # 5-Fold Stratified CV, Overfitting/Underfitting & Confusion Matrix
│   ├── src/unsupervised_learning.py    # K-Means (Elbow), Hierarchical (Dendrogram), DBSCAN
│   ├── src/association_and_pca.py      # Apriori & FP-Growth Market Basket, Course Recommender, PCA
│   ├── src/reinforcement_learning.py   # Student MDP Environment, Q-Learning Agent, Optimal Policy, Ethics
│   ├── src/preprocessing.py           # Missing value imputation, IQR outlier capping, scaling
│   ├── src/feature_engineering.py      # Composite scoring: AcademicScore, EmployabilityIndex, etc.
│   ├── src/resume_scoring.py           # ATS resume keyword matching & feedback
│   ├── src/interview_scoring.py        # Technical/Behavioral interview rubric evaluation
│   └── src/recommendation_engine.py    # 4-Week Personalized Improvement Roadmap generator
│
├── 🧪 AUTOMATED TEST RUNNERS:
│   ├── test_full_syllabus_suite.py     # Master test runner asserting all 7 CE-2 syllabus phases pass
│   └── test_all_curriculum_modules.py  # Comprehensive regression and classification test suite
│
├── 📊 DATASETS (80/20 Split):
│   ├── dataset/train.csv               # 40,000 rows (80%) with Placement_Status & Salary_LPA
│   ├── dataset/test.csv                # 10,000 rows (20%) with Placement_Status & Salary_LPA
│   └── dataset/cleaned_train.csv       # 40,000 rows preprocessed training set
│
├── 💾 SAVED TRAINED ARTIFACTS:
│   ├── models/best_classifier.joblib   # Best classifier (Random Forest, 100% test accuracy)
│   ├── models/best_regressor.joblib    # Best regressor (Multiple Linear Regression, R² = 0.9912)
│   ├── models/cluster_model.joblib     # K-Means clustering model
│   ├── models/scaler.joblib            # Fitted StandardScaler
│   └── models/encoders.joblib          # Categorical LabelEncoders
│
└── 📈 VISUALIZATION PLOTS & REPORTS (outputs/):
    ├── outputs/regression_actual_vs_predicted.png
    ├── outputs/regression_residuals.png
    ├── outputs/overfitting_underfitting_curve.png
    ├── outputs/bias_variance_tradeoff.png
    ├── outputs/detailed_confusion_matrix.png
    ├── outputs/hierarchical_dendrogram.png
    ├── outputs/cluster_elbow_silhouette.png
    ├── outputs/pca_variance_and_projection.png
    ├── outputs/rl_qlearning_convergence.png
    ├── outputs/model_comparison.png
    ├── outputs/roc_curves_comparison.png
    └── outputs/correlation_heatmap.png
```

---

## 🛠️ Step 1: Environment Setup & Activation

### 1.1 Verify Python Version
Ensure you are using Python 3.10 to 3.15:
```powershell
python --version
# Target: Python 3.10+ (Current system configured with Python 3.15 at C:\Python315\python.exe)
```

### 1.2 Navigate to the Project Root
```powershell
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI
```

### 1.3 Choose How You Run Python Commands

#### Option A: Using Activated Virtual Environment (`venv`)
```powershell
# On Windows PowerShell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# On Windows Command Prompt (CMD):
venv\Scripts\activate.bat

# On Git Bash:
source venv/Scripts/activate

# Install required packages (if needed):
pip install -r requirements.txt
```

#### Option B: Using Direct Python Binary (Fastest, No Venv Activation Required)
```powershell
C:\Python315\python.exe -m pip install -r requirements.txt
```

> **Required libraries included:** `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `mlxtend`, `nbformat`, `jsonschema`.

---

## 💻 Step 2: Running the Command-Line Pipeline (`src/predict.py`)

`predict.py` is the central operational pipeline for PlacementIQ AI.

### 2.1 Interactive Student Prediction (Viva Demo Favorite)
Prompts for student details directly in the terminal, computes composite features, and outputs both **Placement Probability** and **Estimated Salary Package in LPA**:
```powershell
python src/predict.py --interactive
```
**Example Inputs to Try:**
- `Age`: 22
- `Gender`: Male
- `Degree`: B.Tech
- `Branch`: CSE
- `CGPA`: 8.7
- `Number of Internships`: 2
- `Number of Projects`: 4
- `Coding Skills`: 8
- `Communication Skills`: 8
- `Aptitude Score`: 85
- `Soft Skills Rating`: 8
- `Number of Certifications`: 3
- `Current Backlogs`: 0

**What It Prints:**
- Placement Probability (e.g. `82.4%`)
- Readiness Assessment (`Placement Ready`)
- Student Cluster (`Top Performers`)
- **Estimated Salary Package / CTC (e.g. `27.85 LPA`)**
- Top contributing features (SHAP-style attribution)
- 4-Week customized improvement roadmap

---

### 2.2 Full End-to-End Training Pipeline
Executes preprocessing, feature engineering, classification training, salary regression, clustering, association rules, PCA, and RL:
```powershell
python src/predict.py --train
```
**What It Does:**
1. Loads 40,000 training records (`dataset/train.csv`) and applies 80/20 train/test evaluation against 10,000 test records (`dataset/test.csv`).
2. Trains all 6 Classifiers and selects the best model (`models/best_classifier.joblib`).
3. Trains Simple Linear, Multiple Linear, and Polynomial Regressors for CTC prediction and saves `models/best_regressor.joblib`.
4. Performs K-Means clustering and segments students into 3 cohorts.
5. Generates high-resolution PNG plots in `outputs/`.

---

### 2.3 Batch Prediction on a CSV File
To predict outcomes for thousands of students at once:
```powershell
python src/predict.py --batch dataset/test.csv
```
Generates `outputs/batch_predictions.csv` with added columns: `Probability`, `Readiness_Level`, and `Cluster`.

---

## 📓 Step 3: Running the 7 Phase Jupyter Notebooks

Every syllabus module has a dedicated, structured notebook with step-by-step code, Markdown explanations, formulas, and visualizations:

| # | Notebook File | Topic Covered | Expected Key Metric / Plot |
|---|---|---|---|
| **Phase 1** | `01_Phase1_Data_Preprocessing_and_EDA.ipynb` | Missing values, IQR outlier capping, 80/20 split, scaling | Boxplots, Correlation Heatmap |
| **Phase 2** | `02_Phase2_Supervised_Regression.ipynb` | Linear, Multiple Linear, Polynomial Regression (Salary CTC) | $R^2 = 0.9912$, $\text{MAE} = 0.3621$ LPA, Residual Plot |
| **Phase 3** | `03_Phase3_Supervised_Classification.ipynb` | Logistic, KNN, Decision Tree, Random Forest, Naive Bayes, SVM | Classification Report, ROC Comparison Plot |
| **Phase 4** | `04_Phase4_Model_Evaluation_and_Validation.ipynb` | 5-Fold Stratified CV, Overfitting/Underfitting Curve, Confusion Matrix | Tree Depth 1–15 validation curves, Annotated Matrix |
| **Phase 5** | `05_Phase5_Unsupervised_Clustering.ipynb` | K-Means, Hierarchical (Ward Dendrogram), DBSCAN | Elbow Curve, Silhouette Score, Dendrogram Plot |
| **Phase 6** | `06_Phase6_Association_Rules_and_Recommendations.ipynb` | Apriori, FP-Growth, Market Basket, Course Recommender, PCA | 149 Itemsets, PCA Scree Plot & 2D Projection |
| **Phase 7** | `07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb` | MDP Environment, Q-Learning Agent, Optimal Policy, AI Ethics | Q-learning Reward Curve, Policy Grid, Comparison Table |

### How to Run the Notebooks:
- **In VS Code:** Open any `.ipynb` file, pick your Python kernel in the top-right corner, and click **"Run All"**.
- **In JupyterLab / Classic Notebook:**
  ```powershell
  jupyter notebook
  ```
- **In Google Colab:** Upload any notebook and the `dataset/train.csv` + `dataset/test.csv` files to Colab and execute.

---

## 🔬 Step 4: Running Individual Python Modules Directly

Each source module in `src/` is modular and can be run independently:

```powershell
# 1. Supervised Learning: Regression Module (Linear, Multiple, Polynomial, MAE, MSE, RMSE, R²)
python src/train_regressor.py

# 2. Supervised Learning: Classification Registry (Logistic, KNN, Tree, Forest, Naive Bayes, SVM)
python src/train_classifier.py

# 3. Model Evaluation & Validation (Stratified K-Fold CV, Overfitting Curves, Confusion Matrix)
python src/model_validation.py

# 4. Unsupervised Learning: Clustering (K-Means, Ward Linkage Dendrogram, DBSCAN)
python src/unsupervised_learning.py

# 5. Association Rule Mining & PCA (Apriori, FP-Growth, Recommender, Scree Plot, 2D Scatter)
python src/association_and_pca.py

# 6. Reinforcement Learning (Student MDP, Bellman Q-Learning Updates, Policy Table, Ethics)
python src/reinforcement_learning.py

# 7. Data Preprocessing & 80/20 Split Verification
python src/preprocessing.py

# 8. Feature Engineering Pipeline
python src/feature_engineering.py

# 9. Rule-Based Resume ATS Scoring
python src/resume_scoring.py

# 10. Technical & Behavioral Interview Scoring
python src/interview_scoring.py

# 11. Student Roadmap & Recommendation Engine
python src/recommendation_engine.py
```

---

## 🧪 Step 5: Running the Automated Verification Test Suites

To verify zero errors across the entire codebase:

### 5.1 Master Syllabus Verification Suite (Recommended for Evaluators)
```powershell
python test_full_syllabus_suite.py
```
**Checks all 7 syllabus phases:**
- ✔ **Phase 1:** 80/20 train/test split verification (40,000 train / 10,000 test).
- ✔ **Phase 2:** Regression training ($R^2 \ge 0.95$, MAE $< 1.0$ LPA).
- ✔ **Phase 3:** All 6 classifiers present and evaluated.
- ✔ **Phase 4:** 5-Fold Stratified CV, Overfitting/Underfitting diagnostics, Confusion matrix.
- ✔ **Phase 5:** K-Means, Hierarchical Dendrogram, and DBSCAN clustering.
- ✔ **Phase 6:** FP-Growth association rules ($\ge 50$ itemsets) and PCA Scree plot.
- ✔ **Phase 7:** Q-Learning reinforcement learning convergence.

### 5.2 Legacy & Curriculum Deep Test Suite
```powershell
python test_all_curriculum_modules.py
```

---

## 📊 Step 6: Visualizations & Generated Artifacts

All generated charts are saved in high resolution (150 DPI) in `outputs/`:

| Output File | What It Illustrates |
|---|---|
| `outputs/regression_actual_vs_predicted.png` | Scatter plot with ideal $y=x$ reference line showing actual vs predicted salary packages. |
| `outputs/regression_residuals.png` | Bell-shaped residual error histogram centered at 0 verifying normal errors. |
| `outputs/overfitting_underfitting_curve.png` | Training vs Validation accuracy across tree depths (1–15) showing the optimal depth. |
| `outputs/bias_variance_tradeoff.png` | U-shaped total error decomposition showing high bias vs high variance regimes. |
| `outputs/detailed_confusion_matrix.png` | Heatmap with labeled True Positives, True Negatives, False Positives, and False Negatives. |
| `outputs/hierarchical_dendrogram.png` | Dendrogram displaying agglomerative hierarchical student clustering with Ward linkage. |
| `outputs/cluster_elbow_silhouette.png` | Inertia (Elbow) curve and Silhouette coefficient curve for optimal $K$. |
| `outputs/pca_variance_and_projection.png` | Explained variance Scree plot + 2D principal component projection of student cohorts. |
| `outputs/rl_qlearning_convergence.png` | Cumulative episode reward convergence curve showing policy optimization. |
| `outputs/model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC across classifiers. |
| `outputs/roc_curves_comparison.png` | Receiver Operating Characteristic (ROC) curves with Area Under Curve (AUC) values. |
| `outputs/correlation_heatmap.png` | Correlation matrix of all academic and skill features. |

---

## ❓ Step 7: Troubleshooting & Common FAQs

### 1. "Activate.ps1 cannot be loaded because running scripts is disabled on this system"
**Cause:** Windows PowerShell script execution policy default is restricted.  
**Fix:** Run this once in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. "ModuleNotFoundError: No module named 'mlxtend'" (or any other package)
**Cause:** The library is not yet installed in the current Python environment.  
**Fix:** Install it via pip:
```powershell
pip install mlxtend
# Or targeting system Python directly:
C:\Python315\python.exe -m pip install mlxtend
```

### 3. "can't open file 'predict.py': No such file or directory"
**Cause:** You are running the command from the wrong folder.  
**Fix:** Always ensure your working directory is `PlacementIQ_AI`:
```powershell
cd d:\01_Projects\01_AimlProject\PlacementIQ_AI
```

### 4. "If you are on Python 3.15 and venv points to an older Python 3.13"
**Fix:** Run commands directly using `C:\Python315\python.exe <script_name>` or recreate the venv:
```powershell
Remove-Item -Recurse -Force venv
C:\Python315\python.exe -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🏆 Summary Checklist for Continuous Evaluation (CE-2)

- [x] **80/20 Train-Test Split:** `train.csv` (40,000 rows) & `test.csv` (10,000 rows) with dual targets (`Placement_Status` and `Salary_LPA`).
- [x] **Regression Module:** Linear, Multiple Linear, Polynomial Regression ($R^2 = 0.9912$, $\text{MAE} = 0.3621$ LPA).
- [x] **Classification Module:** All 6 classifiers (Logistic, KNN, Tree, Forest, Naive Bayes, SVM) benchmarked.
- [x] **Validation & Evaluation:** 5-Fold Stratified CV, Overfitting/Underfitting curve, Bias-Variance curve, Confusion Matrix.
- [x] **Unsupervised Clustering:** K-Means (Elbow & Silhouette), Hierarchical Clustering (Ward Linkage Dendrogram), DBSCAN.
- [x] **Association Rules & PCA:** Apriori & FP-Growth mining, Course Recommender, PCA Scree plot & 2D projection.
- [x] **Reinforcement Learning & Ethics:** MDP Environment, Q-Learning agent, optimal policy table, comparison matrix, AI ethics.
- [x] **7 Standalone Jupyter Notebooks:** One dedicated notebook for each phase.
- [x] **Master Test Suite:** Verified passing with 100% success via `python test_full_syllabus_suite.py`.

