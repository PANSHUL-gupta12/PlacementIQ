# PlacementIQ AI — Project Overview

## 1. What is PlacementIQ AI?

**PlacementIQ AI** is an end-to-end Artificial Intelligence system that predicts a student's **placement readiness** based on their academic, technical, extracurricular, and professional profile.

Unlike simple yes/no classifiers, PlacementIQ AI provides a **comprehensive, multi-dimensional analysis**:

| Feature | Description |
|---------|-------------|
| **Placement Probability** | Exact percentage chance of getting placed (0–100%) |
| **Readiness Level** | Categorical assessment: Beginner → Improving → Competitive → Highly Competitive → Exceptional |
| **Student Clustering** | Groups students into behavioural clusters (e.g., "Top Performers", "Needs Improvement") using unsupervised learning |
| **Explainable AI (XAI)** | SHAP-based explanations showing which factors help or hurt placement chances |
| **Skill Gap Analysis** | Identifies weak areas across 6 skill categories |
| **4-Week Improvement Roadmap** | Personalised weekly action plan to improve weaknesses |
| **Company Matching** | Maps students to realistic company tiers (Mass Recruiters → Dream Companies) |
| **Resume Analysis** | ATS keyword scoring + LLM-based resume classification |
| **Interview Readiness** | Weighted evaluation of technical, communication, and confidence scores |

---

## 2. Problem Statement

Every year, millions of engineering students in India prepare for campus placements. Most students:
- Don't know where they stand compared to placed students
- Can't identify which specific skills they need to improve
- Don't have personalised guidance on what to work on

**PlacementIQ AI solves this** by training machine learning models on historical placement data (50,000 student records) to learn the patterns that separate placed students from unplaced ones, and then providing actionable, personalised insights.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PlacementIQ AI                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐           │
│  │ cleaned_train.csv │──►│ Data Loader  │──►│ Feature       │           │
│  │ test.csv  │   │ + EDA        │   │ Engineering   │           │
│  └──────────┘   └──────────────┘   └──────┬────────┘           │
│                                           │                     │
│                                    ┌──────▼────────┐           │
│                                    │ Preprocessing  │           │
│                                    │ (Scale+Encode) │           │
│                                    └──────┬────────┘           │
│                          ┌────────────────┼────────────────┐    │
│                          │                │                │    │
│                   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──┐│
│                   │ Classifiers │  │  Clustering  │  │  SHAP  ││
│                   │ (5 models)  │  │  (KMeans)    │  │  (XAI) ││
│                   └──────┬──────┘  └──────┬──────┘  └─────┬──┘│
│                          │                │                │    │
│                   ┌──────▼────────────────▼────────────────▼──┐│
│                   │          Prediction Engine                 ││
│                   │    (predict_single_student)                ││
│                   └──────┬──────────────────────────────────┘  │
│                          │                                      │
│        ┌─────────────────┼─────────────────────────┐           │
│        │                 │                         │           │
│  ┌─────▼─────┐   ┌──────▼──────┐   ┌──────────────▼──┐       │
│  │ Skill Gap  │   │  Company    │   │ Resume + LLM    │       │
│  │ + Roadmap  │   │  Matching   │   │ Analysis        │       │
│  └─────┬─────┘   └──────┬──────┘   └──────────┬──────┘       │
│        │                 │                      │              │
│        └─────────────────┼──────────────────────┘              │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │   Report    │                               │
│                   │  Generator  │                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **ML Framework** | scikit-learn | Classical ML models (Random Forest, Logistic Regression, etc.) |
| **Gradient Boosting** | XGBoost | High-performance gradient boosted trees |
| **Deep Learning** | PyTorch + HuggingFace Transformers | DistilBERT fine-tuning for NLP |
| **Explainable AI** | SHAP | Feature importance and prediction explanations |
| **Data Processing** | Pandas, NumPy | Data manipulation and numerical computing |
| **Visualisation** | Matplotlib, Seaborn | EDA plots and model comparison charts |
| **Serialisation** | joblib | Model persistence (save/load trained models) |

---

## 5. Dataset

The system is trained on **50,000 student records** split into:
- `cleaned_train.csv` — 40,000 records for training
- `test.csv` — 10,000 records for evaluation

### Features (13 input columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Age` | Integer | 18–30 | Student's age |
| `Gender` | Categorical | Male/Female | Gender |
| `Degree` | Categorical | B.Tech/B.Sc/BCA/MCA | Degree program |
| `Branch` | Categorical | CSE/IT/ECE/ME/Civil | Engineering branch |
| `CGPA` | Float | 4.0–10.0 | Cumulative GPA |
| `Internships` | Integer | 0–5 | Number of internships |
| `Projects` | Integer | 0–10 | Number of projects |
| `Coding_Skills` | Integer | 0–10 | Self-rated coding ability |
| `Communication_Skills` | Integer | 0–10 | Communication ability |
| `Aptitude_Test_Score` | Integer | 0–100 | Aptitude exam score |
| `Soft_Skills_Rating` | Integer | 0–10 | Soft skills rating |
| `Certifications` | Integer | 0–10 | Number of certifications |
| `Backlogs` | Integer | 0–5 | Active backlogs |

### Target Variables
| Column | Type | Description | Purpose |
|--------|------|-------------|---------|
| `Placement_Status` | Binary ("Placed"/"Not Placed") | Placement outcome | Classification models |
| `Salary_LPA` | Continuous Float (3.0–35.0 LPA) | Expected package | Regression models |

---

## 6. Key Modules

| Module | File | Purpose |
|--------|------|---------|
| Data Loader | `data_loader.py` | Load CSV, detect columns, generate EDA plots |
| Preprocessing | `preprocessing.py` | Handle missing values, encode, scale, and split |
| Feature Engineering | `feature_engineering.py` | Create composite scores and indices |
| Classifier Training | `train_classifier.py` | Train Logistic, KNN, Tree, Forest, Naive Bayes, SVM |
| Salary Regressor | `train_regressor.py` | Simple, Multiple & Polynomial Regression for LPA |
| Model Validation | `model_validation.py` | Bias-Variance, Overfitting curves, Confusion Matrix |
| Unsupervised Clustering | `unsupervised_learning.py` | K-Means, Hierarchical Dendrogram, DBSCAN |
| Association Rules & PCA | `association_and_pca.py` | Apriori, FP-Growth, Recommendations, PCA Scree/2D |
| Reinforcement Learning | `reinforcement_learning.py` | MDP Environment, Q-Learning Agent & Optimal Policy |
| Preprocessing | `preprocessing.py` | Handle missing values, encode categoricals, scale features |
| Feature Engineering | `feature_engineering.py` | Create composite scores (AcademicScore, TechnicalScore, etc.) |
| Classifier Training | `train_classifier.py` | Train 5 ML models, cross-validate, select best |
| Model Evaluation | `evaluate_classifier.py` | Compute metrics, generate comparison plots |
| Clustering | `train_cluster.py` | KMeans clustering with silhouette analysis |
| Explainable AI | `explain_prediction.py` | SHAP values, feature importance plots |
| Recommendation Engine | `recommendation_engine.py` | Skill gap analysis, 4-week roadmap |
| Company Matching | `company_matching.py` | Map probability to company tiers |
| Resume Scoring | `resume_scoring.py` | ATS keyword scoring |
| LLM Resume Classifier | `llm_resume_classifier.py` | Fine-tuned DistilBERT for resume classification |
| Interview Scoring | `interview_scoring.py` | Weighted interview evaluation |
| Report Generator | `report_generator.py` | Generate final placement report |
| Orchestrator | `predict.py` | Main entry point, CLI interface |
