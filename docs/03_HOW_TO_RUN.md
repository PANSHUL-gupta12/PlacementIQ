# How to Run PlacementIQ AI — Complete Setup Guide

This guide takes you from a fresh clone to a fully running AI pipeline. Follow every step carefully.

---

## Prerequisites

Before starting, make sure you have:

| Requirement | Version | Check command |
|-------------|---------|---------------|
| Python | 3.10 or higher | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| Disk Space | ~3 GB | For venv + PyTorch |

---

## Step 1: Navigate to the Project

Open your terminal (Git Bash, PowerShell, or Command Prompt) and navigate to the project folder:

```bash
cd d:/01_Projects/01_AimlProject/PlacementIQ_AI
```

---

## Step 2: Create a Virtual Environment

A virtual environment keeps all project packages isolated from your system Python.

```bash
# Create the virtual environment
python -m venv venv
```

---

## Step 3: Activate the Virtual Environment

**Git Bash (MINGW64):**
```bash
source venv/Scripts/activate
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

✅ You should see `(venv)` at the beginning of your terminal prompt. If you don't, the environment is not active — go back and try again.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: numpy, pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, scipy, transformers, torch, datasets.

> ⚠️ **PyTorch is ~2 GB.** This step may take 5–15 minutes depending on your internet speed.

---

## Step 5: Verify Datasets

Make sure `train.csv` and `test.csv` are in the `dataset/` folder:

```bash
ls dataset/
# Should show: train.csv  test.csv  generate_dataset.py
```

If they're missing, copy them from the parent directory:
```bash
cp ../train.csv dataset/
cp ../test.csv dataset/
```

---

## Step 6: Train the Classical ML Models

This trains 5 classifiers (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, XGBoost), performs clustering, and generates SHAP explanations:

```bash
python src/predict.py --train
```

**Expected runtime:** 5–10 minutes on a standard laptop.

**What happens:**
1. Loads and cleans 45,000 training records
2. Engineers 7 composite features
3. Trains and cross-validates 5 ML models
4. Selects the best model (by ROC-AUC)
5. Clusters all students using KMeans
6. Generates SHAP feature importance plots
7. Saves all models to `models/` and outputs to `outputs/`

---

## Step 7: Train the LLM (DistilBERT)

This fine-tunes a pre-trained DistilBERT transformer on resume-to-placement classification:

```bash
python src/predict.py --train-llm
```

**Expected runtime:**
- **CPU only:** 15–30 minutes
- **With NVIDIA GPU:** 3–5 minutes

**What happens:**
1. Loads `train.csv` and converts each student's stats into a natural language paragraph
2. Tokenises paragraphs using the DistilBERT tokenizer
3. Fine-tunes the model for 3 epochs
4. Saves the trained model to `models/llm_resume_model/`

---

## Step 8: Run Interactive Prediction

This is the main feature! The AI asks you for a student's details and generates a personalised report:

```bash
python src/predict.py --interactive
```

You will be prompted:
```
Age (e.g., 22): 21
Gender (Male/Female): Female
Degree (e.g., B.Tech): B.Tech
Branch (e.g., CSE): CSE
CGPA (0-10): 8.2
Number of Internships: 2
Number of Projects: 5
Coding Skills (0-10): 8
Communication Skills (0-10): 7
Aptitude Score (0-100): 78
Soft Skills Rating (0-10): 7
Number of Certifications: 3
Current Backlogs: 0
```

The AI will instantly generate a full placement report with:
- Placement probability percentage
- Readiness level
- Student cluster
- Strengths and weaknesses
- Company recommendations
- 4-week improvement roadmap

---

## Step 9: Run Batch Prediction

Predict placement for an entire CSV file of students:

```bash
python src/predict.py --batch dataset/test.csv
```

**Output:** A new file `outputs/batch_predictions.csv` with three new columns added:
- `Probability` — Placement probability (%)
- `Readiness_Level` — Categorical assessment
- `Cluster` — Student cluster name

---

## Command Reference

| Command | What it does |
|---------|-------------|
| `python src/predict.py --train` | Train all classical ML models |
| `python src/predict.py --train-llm` | Fine-tune the DistilBERT LLM |
| `python src/predict.py --interactive` | Interactive single-student prediction |
| `python src/predict.py --batch <file.csv>` | Batch prediction on a CSV file |
| `python src/predict.py --help` | Show all available commands |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'joblib'"
**Cause:** Virtual environment is not activated.
**Fix:** Run `source venv/Scripts/activate` (Bash) or `venv\Scripts\activate` (CMD).

### "can't open file 'src/predict.py': No such file or directory"
**Cause:** You are in the wrong folder.
**Fix:** Make sure you `cd PlacementIQ_AI` first.

### "bash: venvScriptsactivate: command not found"
**Cause:** Using backslashes in Git Bash.
**Fix:** Use forward slashes: `source venv/Scripts/activate`

### "CUDA not available" warning during LLM training
**Cause:** You don't have an NVIDIA GPU or CUDA is not installed.
**Impact:** Training still works, just slower (CPU mode). This is perfectly fine.

### UnicodeEncodeError when printing report
**Cause:** Windows terminal can't render Unicode characters.
**Fix:** This is automatically handled. If you still see issues, run:
```bash
chcp 65001
```

---

## Output Files

After running the full pipeline, you'll find these files:

### models/ (Saved AI Models)
| File | Description |
|------|-------------|
| `best_classifier.joblib` | Best ML model (Random Forest) |
| `scaler.joblib` | StandardScaler for feature normalisation |
| `encoders.joblib` | LabelEncoders for categorical features |
| `cluster_model.joblib` | Trained KMeans cluster model |
| `cluster_names.joblib` | Cluster label mapping |
| `feature_names.joblib` | Ordered list of feature column names |
| `llm_resume_model/` | Fine-tuned DistilBERT model + tokenizer |

### outputs/ (Reports and Visualisations)
| File | Description |
|------|-------------|
| `placement_report.txt` | Latest generated placement report |
| `batch_predictions.csv` | Batch prediction results |
| `model_comparison.png` | Bar chart comparing model AUCs |
| `model_comparison.csv` | Tabular model comparison |
| `feature_importance.png` | Feature importance bar chart |
| `shap_summary.png` | SHAP beeswarm plot |
| `correlation_heatmap.png` | Feature correlation matrix |
| `eda_*.png` | Various EDA visualisation plots |
| `cleaned_data.csv` | Preprocessed and cleaned dataset |
