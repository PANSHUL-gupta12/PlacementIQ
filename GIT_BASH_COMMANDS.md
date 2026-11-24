# PlacementIQ AI — Verified Master Git Bash Commands Guide

All commands in this guide are **100% verified inside Git Bash (MinGW64 / MSYS2)** on Windows.
Do **not** use Windows Command Prompt (`cmd`) backslashes (`\`) or `.bat` scripts inside Git Bash—always use **forward slashes (`/`)**.

---

## 0. Is My Model Already Trained?

**YES — All Machine Learning, Regression, Clustering, and LLM models are already trained and saved in `models/`.**
You can run predictions (`--interactive` and `--batch`) immediately without waiting to retrain.

| Model Artifact | File / Folder | Trained Algorithm & Verified Performance |
|---|---|---|
| **Placement Classifier** | `models/best_classifier.joblib` | **RandomForestClassifier** (Selected best of 8 classifiers — Test ROC-AUC: **1.0000**, 5-Fold CV AUC: **1.0000**) |
| **Salary Package Regressor** | `models/best_regressor.joblib` | **Multiple Linear Regression** (Trained on 40,000 records — R²: **0.9910**, MAE: **0.3646 LPA**, RMSE: **0.4581 LPA**) |
| **Student Clustering** | `models/cluster_model.joblib` | **K-Means Clustering** (Segments students into *Top Performers* and *Needs Improvement*) |
| **Feature Scaler & Encoders** | `models/scaler.joblib`, `models/encoders.joblib`, `models/feature_names.joblib` | Fitted `StandardScaler` & `LabelEncoder` across 20 features (`Salary_LPA` properly excluded to prevent leakage) |
| **NLP Resume Classifier (LLM)** | `models/llm_resume_model/` | Fine-tuned **DistilBERT** (`model.safetensors`, ~267 MB — verified **Placed** confidence **99.04%** on strong profiles) |

Check your saved model files anytime in Git Bash:

```bash
ls -lh models/ models/llm_resume_model/
```

---

## 1. Git Bash Startup & Activation (Always Run First)

Every time you open a fresh **Git Bash** window, copy-paste these lines first:

```bash
# 1. Navigate to the project folder (Git Bash uses /d/ instead of D:\)
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI

# 2. Activate the Python virtual environment
source venv/Scripts/activate

# 3. Verify your prompt shows (venv) and Python resolves to the project venv
which python
python --version
```

> **Important for Git Bash:** Always use `source venv/Scripts/activate` (never `venv\Scripts\activate` or `activate.bat`).

---

## 2. Main Prediction Commands (Interactive & Batch)

### A. Interactive Single-Student Prediction (Prompts in Terminal)

```bash
# Run interactive single-student prediction (Placement % + Package LPA + LLM + 4-Week Roadmap)
python src/predict.py --interactive
```

> **Git Bash MinTTY Tip:** If you open standalone **Git Bash** (not VS Code's terminal) and prompts do not appear immediately while waiting for keyboard input, launch it with `winpty`:
> ```bash
> winpty python src/predict.py --interactive
> ```

**Quick Test Inputs to Try When Prompted** (or simply press `Enter` on any prompt to accept the default):
- `Age`: `22`
- `Gender`: `Male` *(or `Female`)*
- `Degree`: `B.Tech` *(`B.Tech`, `B.Sc`, `BCA`, `MCA`)*
- `Branch`: `CSE` *(`CSE`, `ECE`, `IT`, `ME`, `Civil`)*
- `CGPA`: `8.7`
- `Number of Internships`: `2`
- `Number of Projects`: `4`
- `Coding Skills (0-10)`: `8`
- `Communication Skills (0-10)`: `8`
- `Aptitude Score (0-100)`: `85`
- `Soft Skills Rating (0-10)`: `8`
- `Number of Certifications`: `3`
- `Current Backlogs`: `0`

Expected output: `Placement Probability: 74.1%`, `Readiness Level: Placement Ready`, `Student Cluster: Top Performers`, `Estimated Package/CTC: 27.18 LPA`, `LLM Prediction: Placed (99.0%)`, and full report saved to `outputs/placement_report.txt`.

---

### B. Batch Prediction on 10,000 Students (Vectorized — Runs in < 1 Second)

Both `--batch` and `--input` / `--output` syntax work cleanly:

```bash
# Standard batch prediction (reads dataset/test.csv -> saves outputs/batch_predictions.csv)
python src/predict.py --batch dataset/test.csv

# OR specify custom input and output paths
python src/predict.py --input dataset/test.csv --output outputs/batch_predictions.csv

# View the first 10 predictions in Git Bash
head -n 11 outputs/batch_predictions.csv
```

Outputs added for every student row: `Probability`, `Readiness_Level`, `Cluster`, and `Estimated_Salary_LPA`.

---

### C. Quick Report Formatter Check

```bash
# Render sample institutional placement readiness report
python src/report_generator.py
```

---

## 3. Automated Test Suites & Core Syllabus Modules (From Viva Checklist)

All 7 syllabus modules now support direct standalone execution from Git Bash:

```bash
# 1. Run Complete Automated Syllabus Test Suite (All 7 Phases in ~15 seconds)
python test_full_syllabus_suite.py

# 2. Run Comprehensive Modular Curriculum Test Suite (Tests 1–4)
python test_all_curriculum_modules.py

# 3. Run Continuous Salary Package Regression Training (Linear, Multiple, Polynomial; R²=0.991)
python src/train_regressor.py

# 4. Run K-Fold Cross-Validation, Confusion Matrix & Learning Curve Diagnostics
python src/model_validation.py

# 5. Run Unsupervised Clustering (K-Means Elbow/Silhouette, Dendrogram, DBSCAN)
python src/unsupervised_learning.py

# 6. Run Apriori & FP-Growth Association Rules, Recommender & PCA
python src/association_and_pca.py

# 7. Run Reinforcement Learning (Q-Learning MDP Policy Convergence Simulator)
python src/reinforcement_learning.py
```

---

## 4. All Other Standalone Python Modules (`src/`)

Every data processing, feature engineering, classifier training, LLM, and scoring module runs cleanly in Git Bash:

```bash
# Exploratory Data Analysis (Generates correlation heatmap, class distribution, histograms, boxplots)
python src/data_loader.py

# Feature Engineering Pipeline (Generates AcademicScore, TechnicalScore, OverallSkillScore, etc.)
python src/feature_engineering.py

# Full Data Preprocessing Pipeline (Encoding, IQR outlier capping, StandardScaler, 80/20 prep)
python src/preprocessing.py

# Train & Benchmark All 8 Classification Models (RandomForest, GradientBoosting, XGBoost, KNN, SVM, Logistic, DT, NB)
python src/train_classifier.py

# Test Saved Fine-Tuned DistilBERT Resume Classifier (Instant inference check)
python src/llm_resume_classifier.py

# Standalone Career Feature Modules
python src/company_matching.py
python src/resume_scoring.py
python src/interview_scoring.py
python src/recommendation_engine.py
```

---

## 5. Full Retraining Commands (Only Run If You Want to Retrain Models)

Since your models are already trained, you do **not** need to run these unless you modify the dataset or code and want fresh weights:

```bash
# Full end-to-end pipeline retraining (~2–3 minutes; regenerates all models, CSVs & charts)
python src/predict.py --train

# Fine-tune DistilBERT Transformer from scratch (~10–25 minutes on CPU; saved model already exists!)
python src/predict.py --train-llm
# OR directly:
python src/llm_resume_classifier.py --train --epochs 3
```

---

## 6. Inspecting Outputs, CSV Metrics & Graphs in Git Bash

```bash
# View salary regression accuracy table (MAE, MSE, RMSE, R2_Score)
cat outputs/regression_metrics_comparison.csv

# View 8-classifier benchmark results
cat outputs/model_comparison.csv

# View top 10 Apriori/FP-Growth placement rules
head -n 11 outputs/placement_association_rules.csv

# View Q-Learning optimal preparation policy
cat outputs/rl_optimal_policy.csv

# View PCA feature loadings
cat outputs/pca_feature_loadings.csv

# View student cluster profile averages
cat outputs/student_cluster_profiles.csv

# View latest single-student placement report
cat outputs/placement_report.txt

# List all generated PNG graphs
ls -lh outputs/*.png

# List all trained model files
ls -lh models/ models/llm_resume_model/
```

---

## 7. Running Jupyter Notebooks (`.ipynb` Files)

All 12 `.ipynb` files are already executed and saved cleanly.
- **Recommended (Works Right Now):** Open any notebook (`01_Phase1_Data_Preprocessing_and_EDA.ipynb` through `07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb`) in **VS Code**, select the **`venv` (`Python 3.15.2`)** kernel at top-right, and click **Run All** (`ipykernel` is already installed).
- **If you want browser JupyterLab / CLI `nbconvert` in Git Bash:** install `jupyterlab` and `nbconvert` first, then launch:

```bash
# Install browser Jupyter tools into venv (only needed if not using VS Code)
python -m pip install jupyterlab notebook nbconvert

# Launch JupyterLab in browser
python -m jupyter lab

# Execute a notebook headlessly from Git Bash
python -m jupyter nbconvert --to notebook --execute 02_Phase2_Supervised_Regression.ipynb --inplace
```

---

## 8. Git & GitHub Workflow Commands (Git Bash)

```bash
# One-time safe directory configuration (if on D: drive)
git config --global --add safe.directory D:/01_Projects/01_AimlProject/PlacementIQ_AI

# Check repository status
git status

# View recent commits
git log --oneline -n 5

# Stage, commit, and push updates to GitHub (main branch)
git add .
git commit -m "Sync updates"
git push origin main
```

---

## 9. Quick Viva / Live Demo One-Liners (Copy-Paste Ready)

```bash
# One-liner 1: Full 7-phase automated test verification (~15 seconds)
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI && source venv/Scripts/activate && python test_full_syllabus_suite.py

# One-liner 2: Instant 10,000-student batch prediction + inspect top 5 rows (< 2 seconds)
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI && source venv/Scripts/activate && python src/predict.py --batch dataset/test.csv && head -n 6 outputs/batch_predictions.csv

# One-liner 3: Interactive live student prediction demo (with winpty fallback for standalone Git Bash)
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI && source venv/Scripts/activate && winpty python src/predict.py --interactive
```

---

## 10. Git Bash Troubleshooting Quick Fixes

| Issue in Git Bash | Cause | Instant Fix in Git Bash |
|---|---|---|
| `bash: venvScriptspython.exe: command not found` | Used backslashes `\` instead of `/` | Always run `source venv/Scripts/activate` then `python ...` (with `/`) |
| `source: venv/Scripts/activate: No such file or directory` | Not in project root folder | Run `cd /d/01_Projects/01_AimlProject/PlacementIQ_AI` first |
| `--interactive` freezes before showing prompts | Standalone MinTTY PTY buffering | Run `winpty python src/predict.py --interactive` |
| `python` points to wrong/old Python version | `venv` wasn't activated | Run `source venv/Scripts/activate` and check `which python` |
| Git says `dubious ownership in repository` | Windows ownership mismatch on `D:` | Run `git config --global --add safe.directory D:/01_Projects/01_AimlProject/PlacementIQ_AI` |
