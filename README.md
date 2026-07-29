<div align="center">

# 🎓 PlacementIQ AI
**AI-Powered Placement Readiness & Career Prediction Platform**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)

*A comprehensive end-to-end Machine Learning system that predicts campus placement readiness, identifies skill gaps, and generates personalised improvement roadmaps.*

</div>

---

## 🌟 Overview

**PlacementIQ AI** goes beyond traditional binary classification (Placed/Not Placed). It provides a multi-dimensional, deeply personalised analysis of an engineering student's profile by combining classical Machine Learning, Deep Learning NLP, and Explainable AI.

Instead of a simple "yes" or "no", the system provides:
1. **Exact Placement Probability** (e.g., 84.5%)
2. **Explainable AI** (SHAP values showing *why* the model made its decision)
3. **Skill Gap Analysis** across 6 distinct categories
4. **4-Week Personalised Roadmap** targeting specific weaknesses
5. **Resume ATS & LLM Analysis** using a fine-tuned DistilBERT transformer
6. **Student Clustering** mapping the student to behavioural peer groups

---

## 📚 Comprehensive Documentation

The project includes an extensive `docs/` folder covering every aspect of the system. **Please read these documents to fully understand the project:**

- 📄 [**01_PROJECT_OVERVIEW.md**](docs/01_PROJECT_OVERVIEW.md) — System architecture, tech stack, and feature list.
- 🧠 [**02_MODEL_TRAINING.md**](docs/02_MODEL_TRAINING.md) — Detailed explanation of the 7 ML/Deep Learning models used.
- 🚀 [**03_HOW_TO_RUN.md**](docs/03_HOW_TO_RUN.md) — Step-by-step setup and execution guide.
- 🧪 [**04_TESTING_GUIDE.md**](docs/04_TESTING_GUIDE.md) — How to test and validate the AI models.
- 💻 [**05_CODE_WALKTHROUGH.md**](docs/05_CODE_WALKTHROUGH.md) — Complete explanation of all 14 Python modules.
- 🎤 [**06_VIVA_QUESTIONS.md**](docs/06_VIVA_QUESTIONS.md) — 60+ viva questions and detailed answers.
- 📊 [**07_PRESENTATION_AND_REPORT.md**](docs/07_PRESENTATION_AND_REPORT.md) — Ready-to-use 12-slide PPT content and report structure.

---

## 🛠️ Quick Start Guide

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate      # On Windows Git Bash
# venv\Scripts\activate           # On Windows Command Prompt

# Install dependencies (includes PyTorch)
pip install -r requirements.txt
```

### 2. Train the AI Models
```bash
# 1. Train the 5 classical ML models + Clustering
python src/predict.py --train

# 2. Fine-tune the DistilBERT LLM on resume text
python src/predict.py --train-llm
```

### 3. Run Predictions
```bash
# Predict for a single student via terminal prompts (Recommended)
python src/predict.py --interactive

# Predict for an entire CSV file
python src/predict.py --batch dataset/test.csv
```

---

## 📁 Project Structure

```text
PlacementIQ_AI/
├── dataset/                  # Contains train.csv (45k) and test.csv (5k)
├── docs/                     # 7 comprehensive documentation files
├── models/                   # Saved .joblib models + llm_resume_model/
├── outputs/                  # Generated SHAP plots, EDA charts, and reports
├── src/                      # 14 modular Python files
│   ├── predict.py            # Main entry point (CLI Orchestrator)
│   ├── feature_engineering.py# Composite feature creation
│   ├── train_classifier.py   # Trains Random Forest, XGBoost, etc.
│   ├── llm_resume_classifier.py # DistilBERT fine-tuning
│   └── ...                   # (See docs/05_CODE_WALKTHROUGH.md)
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## 🧠 Core Technologies

| Category | Technology |
|----------|------------|
| **Classification** | Random Forest, Gradient Boosting, XGBoost, Decision Tree, Logistic Regression |
| **Clustering** | KMeans (Unsupervised Learning) |
| **NLP (LLM)** | PyTorch, HuggingFace Transformers (DistilBERT) |
| **Explainable AI** | SHAP (SHapley Additive exPlanations) |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |
| **Visualisation** | Matplotlib, Seaborn |

---

<div align="center">
<i>Designed and developed as a complete end-to-end AI/ML placement solution.</i>
</div>
