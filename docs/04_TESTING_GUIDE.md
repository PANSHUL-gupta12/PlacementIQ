# Testing Guide — How to Verify PlacementIQ AI

This document explains how to test every component of the system and validate that the AI models are working correctly.

---

## Table of Contents
1. [Testing Strategy Overview](#1-testing-strategy-overview)
2. [ML Model Testing](#2-ml-model-testing)
3. [LLM Model Testing](#3-llm-model-testing)
4. [Interactive Mode Testing](#4-interactive-mode-testing)
5. [Batch Mode Testing](#5-batch-mode-testing)
6. [Component-Level Testing](#6-component-level-testing)
7. [Sample Test Cases](#7-sample-test-cases)

---

## 1. Testing Strategy Overview

PlacementIQ AI uses the following testing methodology:

| Test Type | What it tests | How |
|-----------|---------------|-----|
| **Train/Test Split** | Model generalisation | 45,000 train + 5,000 test (pre-split) |
| **Cross-Validation** | Model stability | 5-fold stratified CV on training set |
| **Metric Evaluation** | Model quality | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Functional Testing** | Pipeline correctness | Run each mode and verify output |
| **Edge Case Testing** | Robustness | Test with extreme/unusual input values |

---

## 2. ML Model Testing

### 2.1 Automated Evaluation (Built-in)

When you run `python src/predict.py --train`, the pipeline automatically:

1. **Trains 5 models** on the 45,000 training samples
2. **Evaluates each model** on the held-out 5,000 test samples
3. **Cross-validates** each model with 5-fold stratified CV
4. **Generates comparison plots** saved to `outputs/model_comparison.png`
5. **Prints a classification report** (precision, recall, F1 for each class)

### 2.2 How to Read the Results

After training, check the console output for:

```
MODEL COMPARISON
──────────────────────────────────────────────
  RandomForest         — Test AUC: 1.0000 | CV AUC: 1.0000 ± 0.0000
  GradientBoosting     — Test AUC: 1.0000 | CV AUC: 1.0000 ± 0.0000
  LogisticRegression   — Test AUC: 0.9398 | CV AUC: 0.9398 ± 0.0022
  DecisionTree         — Test AUC: 1.0000 | CV AUC: 1.0000 ± 0.0000
  XGBoost              — Test AUC: 1.0000 | CV AUC: 1.0000 ± 0.0000
```

**What to check:**
- AUC should be > 0.90 for a good model
- CV standard deviation should be small (< 0.01 means stable)
- Best model is automatically selected and saved

### 2.3 Verify Saved Models

After training, check that all model files exist:

```bash
ls models/
# Expected output:
# best_classifier.joblib
# cluster_model.joblib
# cluster_names.joblib
# encoders.joblib
# feature_names.joblib
# scaler.joblib
```

### 2.4 Verify Output Plots

```bash
ls outputs/*.png
# Expected:
# correlation_heatmap.png
# eda_box_*.png
# eda_count_*.png
# eda_distribution_*.png
# feature_importance.png
# model_comparison.png
# shap_summary.png
```

---

## 3. LLM Model Testing

### 3.1 Training Verification

After running `python src/predict.py --train-llm`, verify:

```bash
ls models/llm_resume_model/
# Expected:
# config.json
# model.safetensors (or pytorch_model.bin)
# tokenizer_config.json
# vocab.txt
# special_tokens_map.json
```

### 3.2 Quick Prediction Test

You can test the LLM directly:

```python
# Run this in Python (with venv activated)
from src.llm_resume_classifier import load_llm_model, predict_resume_llm

model, tokenizer, device = load_llm_model("models/llm_resume_model")
result = predict_resume_llm(
    "A 22 year old B.Tech CSE student with CGPA 9.0, 3 internships, "
    "5 projects, coding skills 9/10, excellent communication.",
    model, tokenizer, device
)
print(result)
# Expected: {"label": "Placed", "confidence": 0.95+}
```

### 3.3 Edge Cases for LLM

Test with these inputs:
- **Empty string**: Should return a low confidence prediction
- **Very long text**: Should be truncated to 128 tokens and still work
- **Non-English text**: Should still produce a prediction (may be inaccurate)
- **Random characters**: Should not crash

---

## 4. Interactive Mode Testing

### 4.1 Normal Test Case

Run `python src/predict.py --interactive` and enter:

```
Age: 22
Gender: Male
Degree: B.Tech
Branch: CSE
CGPA: 8.5
Internships: 2
Projects: 5
Coding Skills: 8
Communication Skills: 7
Aptitude Score: 80
Soft Skills Rating: 7
Certifications: 3
Backlogs: 0
```

**Expected:** Placement probability > 70%, Readiness Level "Competitive" or higher.

### 4.2 Low Profile Test Case

```
Age: 20
Gender: Female
Degree: BCA
Branch: IT
CGPA: 5.0
Internships: 0
Projects: 1
Coding Skills: 3
Communication Skills: 4
Aptitude Score: 35
Soft Skills Rating: 3
Certifications: 0
Backlogs: 3
```

**Expected:** Placement probability < 30%, Readiness Level "Beginner".

### 4.3 Perfect Profile Test Case

```
Age: 23
Gender: Male
Degree: B.Tech
Branch: CSE
CGPA: 9.5
Internships: 3
Projects: 8
Coding Skills: 10
Communication Skills: 9
Aptitude Score: 95
Soft Skills Rating: 9
Certifications: 5
Backlogs: 0
```

**Expected:** Placement probability > 95%, Readiness Level "Exceptional".

### 4.4 Edge Cases

| Input | Expected Behaviour |
|-------|-------------------|
| Empty (just press Enter) | Uses default values |
| CGPA = 0 | Should work, very low probability |
| Backlogs = 5 | Should significantly reduce probability |
| Gender = "Other" | Falls back to 0 (unknown category) |
| Non-numeric for Age | Shows error message and exits |

---

## 5. Batch Mode Testing

### 5.1 Test with the Test Dataset

```bash
python src/predict.py --batch dataset/test.csv
```

**Expected:**
- Creates `outputs/batch_predictions.csv`
- File has 5,000 rows + 3 new columns (Probability, Readiness_Level, Cluster)
- No errors or crashes

### 5.2 Verify Output

```python
import pandas as pd
df = pd.read_csv("outputs/batch_predictions.csv")
print(df.shape)                    # Should be (5000, 18)
print(df["Probability"].describe())  # Stats of predictions
print(df["Readiness_Level"].value_counts())  # Distribution of readiness levels
print(df["Cluster"].value_counts())  # Distribution across clusters
```

### 5.3 Test with Custom CSV

Create a small custom CSV with known profiles to verify predictions make sense:

```csv
Student_ID,Age,Gender,Degree,Branch,CGPA,Internships,Projects,Coding_Skills,Communication_Skills,Aptitude_Test_Score,Soft_Skills_Rating,Certifications,Backlogs,Placement_Status
1,22,Male,B.Tech,CSE,9.0,3,6,9,8,90,8,4,0,Placed
2,20,Female,BCA,Civil,5.0,0,1,2,3,30,3,0,4,Not Placed
```

---

## 6. Component-Level Testing

### 6.1 Data Loader

```python
from src.data_loader import load_dataset, detect_columns
df = load_dataset("dataset/train.csv")
info = detect_columns(df)
print(info["target"])       # Should print "Placement_Status"
print(len(info["numerical"]))  # Should be > 10
print(info["categorical"])   # Should include "Gender", "Degree", "Branch"
```

### 6.2 Feature Engineering

```python
from src.data_loader import load_dataset
from src.feature_engineering import feature_engineering_pipeline
df = load_dataset("dataset/train.csv")
df_eng = feature_engineering_pipeline(df)
assert "AcademicScore" in df_eng.columns
assert "TechnicalScore" in df_eng.columns
assert "OverallSkillScore" in df_eng.columns
print("✅ Feature engineering passed")
```

### 6.3 Resume Scoring

```python
from src.resume_scoring import score_resume
result = score_resume("Python developer with 3 years experience in machine learning and Docker")
assert "ats_score" in result
assert result["ats_score"] > 0
print(f"✅ ATS Score: {result['ats_score']}%")
```

### 6.4 Interview Scoring

```python
from src.interview_scoring import evaluate_interview
result = evaluate_interview(
    technical_score=80,
    communication_score=70,
    confidence_score=75,
    question_accuracy=85
)
assert "overall_score" in result
print(f"✅ Interview Score: {result['overall_score']}")
```

---

## 7. Sample Test Cases

### Test Matrix

| # | CGPA | Coding | Projects | Internships | Backlogs | Expected Probability | Expected Level |
|---|------|--------|----------|-------------|----------|---------------------|----------------|
| 1 | 9.5 | 10 | 8 | 3 | 0 | >90% | Exceptional |
| 2 | 8.0 | 7 | 4 | 1 | 0 | 50-80% | Competitive |
| 3 | 6.5 | 5 | 2 | 0 | 1 | 20-50% | Improving |
| 4 | 5.0 | 3 | 1 | 0 | 3 | <20% | Beginner |
| 5 | 7.5 | 8 | 5 | 2 | 0 | 60-85% | Competitive |

### Validation Rules

The following should ALWAYS be true:
1. Higher CGPA → Higher probability (all else being equal)
2. More backlogs → Lower probability
3. More internships/projects → Higher probability
4. The probability should be between 0 and 100
5. The readiness level should correspond to the probability
6. The report should always generate without errors
