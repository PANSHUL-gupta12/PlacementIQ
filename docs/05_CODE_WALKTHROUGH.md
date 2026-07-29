# Code Walkthrough — Understanding Every Python File

This document explains every Python file in the project, function by function, so you can fully understand what the code does.

---

## Table of Contents
1. [Project Structure](#1-project-structure)
2. [predict.py — The Main Orchestrator](#2-predictpy--the-main-orchestrator)
3. [data_loader.py — Loading & EDA](#3-data_loaderpy--loading--eda)
4. [feature_engineering.py — Creating Smart Features](#4-feature_engineeringpy--creating-smart-features)
5. [preprocessing.py — Cleaning & Transforming Data](#5-preprocessingpy--cleaning--transforming-data)
6. [train_classifier.py — Training 5 ML Models](#6-train_classifierpy--training-5-ml-models)
7. [evaluate_classifier.py — Measuring Model Quality](#7-evaluate_classifierpy--measuring-model-quality)
8. [train_cluster.py — Grouping Students](#8-train_clusterpy--grouping-students)
9. [explain_prediction.py — Making AI Transparent](#9-explain_predictionpy--making-ai-transparent)
10. [recommendation_engine.py — Skill Gaps & Roadmaps](#10-recommendation_enginepy--skill-gaps--roadmaps)
11. [company_matching.py — Mapping to Companies](#11-company_matchingpy--mapping-to-companies)
12. [resume_scoring.py — ATS Resume Analysis](#12-resume_scoringpy--ats-resume-analysis)
13. [interview_scoring.py — Interview Evaluation](#13-interview_scoringpy--interview-evaluation)
14. [report_generator.py — Building the Final Report](#14-report_generatorpy--building-the-final-report)
15. [llm_resume_classifier.py — DistilBERT Fine-Tuning](#15-llm_resume_classifierpy--distilbert-fine-tuning)
16. [Data Flow Diagram](#16-data-flow-diagram)

---

## 1. Project Structure

```
PlacementIQ_AI/
├── src/                          # All Python source code
│   ├── predict.py                # Main entry point (CLI)
│   ├── data_loader.py            # Dataset loading + EDA charts
│   ├── feature_engineering.py    # Composite feature creation
│   ├── preprocessing.py          # Data cleaning + encoding + scaling
│   ├── train_classifier.py       # Train 5 ML models
│   ├── evaluate_classifier.py    # Model comparison metrics
│   ├── train_cluster.py          # KMeans clustering
│   ├── explain_prediction.py     # SHAP + feature importance
│   ├── recommendation_engine.py  # Skill gap analysis + roadmap
│   ├── company_matching.py       # Company tier matching
│   ├── resume_scoring.py         # ATS keyword scoring
│   ├── interview_scoring.py      # Interview evaluation
│   ├── report_generator.py       # Final report formatter
│   └── llm_resume_classifier.py  # DistilBERT fine-tuning
├── dataset/
│   ├── train.csv                 # 45,000 training records
│   └── test.csv                  # 5,000 test records
├── models/                       # Saved trained models
├── outputs/                      # Generated plots + reports
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # Project readme
```

---

## 2. predict.py — The Main Orchestrator

**Purpose:** This is the entry point of the entire application. It ties all other modules together and provides the CLI interface.

### Key Functions

#### `main()`
The entry point. Parses command-line arguments and routes to the appropriate mode:
```
--train        → run_training_pipeline()
--train-llm    → Trains the DistilBERT LLM model
--interactive  → run_interactive_prediction()
--batch <file> → run_batch_prediction(file)
```

#### `run_training_pipeline()`
Executes the full training process in 7 sequential steps:
1. Calls `run_eda()` to generate EDA visualisations
2. Calls `feature_engineering_pipeline()` on both train and test data
3. Calls `preprocess_pipeline()` to clean, encode, and scale
4. Calls `train_all_models()` to train 5 classifiers
5. Calls `evaluation_pipeline()` to compute metrics and plots
6. Calls `train_cluster_pipeline()` to run KMeans
7. Calls `explanation_pipeline()` to generate SHAP values

#### `predict_single_student(student_data, ...)`
The core prediction function. Takes a dictionary of raw student features and:
1. Applies feature engineering (creates composite scores inline)
2. Encodes categorical values using saved LabelEncoders
3. Builds a feature DataFrame in the correct column order
4. Scales using the saved StandardScaler
5. Predicts placement probability using the best classifier
6. Assigns a student cluster using KMeans
7. Generates SHAP explanation
8. Runs skill gap analysis and company matching
9. Optionally scores resume text and interview responses
10. Returns a complete formatted report string

#### `run_interactive_prediction()`
Prompts the user for each feature value one by one using `input()`. Validates the inputs, then calls `predict_single_student()`.

#### `run_batch_prediction(input_csv)`
Loads an entire CSV, processes each row through the same feature engineering + prediction logic, and saves results with Probability, Readiness_Level, and Cluster columns.

---

## 3. data_loader.py — Loading & EDA

**Purpose:** Load CSV files, auto-detect column types, and generate exploratory data analysis visualisations.

### Key Functions

#### `load_dataset(filepath) → DataFrame`
Simply reads a CSV file and returns a Pandas DataFrame.

#### `detect_columns(df) → dict`
Automatically identifies:
- **Target column**: Finds it by keyword matching ("placed", "status", "target") or by finding the last binary column
- **Categorical columns**: Columns with `object` or `category` dtype
- **Numerical columns**: Columns with numeric dtype
- Removes `Student_ID` and target from feature lists

#### `run_eda(filepath, output_dir) → (DataFrame, dict)`
Generates comprehensive EDA charts:
- Distribution plots for every numerical feature
- Count plots for every categorical feature
- Box plots by target class
- Correlation heatmap
- Missing value report

---

## 4. feature_engineering.py — Creating Smart Features

**Purpose:** Creates 7 composite features that capture higher-level student attributes the raw data doesn't express directly.

### The COMPOSITE_FEATURES Dictionary

This is the heart of feature engineering. Each entry defines:
- **columns**: Which raw columns to combine
- **weights**: How much each column contributes

```python
COMPOSITE_FEATURES = {
    "AcademicScore": {
        "columns": ["CGPA", "Backlogs"],
        "weights": [0.7, -0.3]   # Backlogs penalise the score
    },
    "TechnicalScore": {
        "columns": ["Coding_Skills", "Aptitude_Test_Score"],
        "weights": [0.6, 0.4]
    },
    ...
}
```

#### `feature_engineering_pipeline(df) → DataFrame`
1. For each composite feature, calculates a weighted normalised sum
2. Normalises all engineered features to the [0, 1] range
3. Creates an `OverallSkillScore` as a master metric
4. Returns the DataFrame with new columns added

---

## 5. preprocessing.py — Cleaning & Transforming Data

**Purpose:** Clean the data, handle missing values, encode categorical variables, detect outliers, and scale features.

### Pipeline Steps

#### `remove_duplicates(df) → DataFrame`
Drops identical rows.

#### `handle_missing_values(df, numerical_cols, categorical_cols) → DataFrame`
- Numerical: Fill with **median** (robust to outliers)
- Categorical: Fill with **mode** (most frequent value)

#### `encode_categoricals(df, categorical_cols) → (DataFrame, dict)`
Uses `LabelEncoder` to convert strings to integers:
- Gender: Female → 0, Male → 1
- Degree: B.Sc → 0, B.Tech → 1, BCA → 2, MCA → 3
- Branch: CSE → 0, Civil → 1, ECE → 2, IT → 3, ME → 4

Saves encoders for later use during prediction.

#### `detect_outliers(df, numerical_cols) → DataFrame`
Uses IQR (Interquartile Range) method:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```
Values below/above bounds are **capped** (not removed).

#### `scale_features(df, numerical_cols) → (DataFrame, StandardScaler)`
Applies StandardScaler: transforms each column to mean=0, std=1.
```
x_scaled = (x - mean) / standard_deviation
```

#### `preprocess_pipeline(df_train, df_test, col_info, output_dir, model_dir) → dict`
Orchestrates all steps above on the concatenated train+test data, then splits back.

---

## 6. train_classifier.py — Training 5 ML Models

**Purpose:** Train, cross-validate, and compare 5 different classification algorithms.

#### `train_all_models(X_train, y_train, X_test, y_test) → dict`
For each of the 5 models:
1. Fits the model on training data
2. Predicts on test data
3. Computes ROC-AUC score
4. Runs 5-fold stratified cross-validation
5. Returns all models and their scores

#### `save_best_model(model, name, model_dir)`
Saves the best-performing model to a `.joblib` file for later use.

---

## 7. evaluate_classifier.py — Measuring Model Quality

**Purpose:** Compute detailed evaluation metrics and generate comparison visualisations.

#### `evaluate_model(model, name, X_test, y_test) → dict`
Computes: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

#### `evaluation_pipeline(all_models, best_name, X_test, y_test, output_dir)`
1. Evaluates all models
2. Generates a bar chart comparing their AUC scores
3. Prints a classification report for the best model
4. Saves comparison table as CSV

---

## 8. train_cluster.py — Grouping Students

**Purpose:** Use unsupervised KMeans clustering to group students by their feature profiles.

#### `train_cluster_pipeline(X, feature_names, output_dir, model_dir) → dict`
1. Tests K=2 through K=10
2. Computes Silhouette Score for each K
3. Selects the K with the highest silhouette
4. Fits final KMeans with optimal K
5. Names clusters by average feature values
6. Saves model and cluster mappings

#### `predict_cluster(X, model, cluster_names) → dict`
Assigns a single student to one of the trained clusters.

---

## 9. explain_prediction.py — Making AI Transparent

**Purpose:** Use SHAP (SHapley Additive exPlanations) to explain what features drive each prediction.

#### `explanation_pipeline(model, X_test, feature_names, output_dir)`
1. Computes feature importance from the Random Forest
2. Generates feature importance bar chart
3. Computes SHAP values for the test set
4. Generates SHAP summary (beeswarm) plot

Uses a robust fallback chain:
- First tries `TreeExplainer` (fast, for tree-based models)
- Falls back to `LinearExplainer` (for logistic regression)
- Falls back to `KernelExplainer` (universal, but slow)

#### `explain_single_prediction(model, X, feature_names) → dict`
Returns which features push the prediction UP (positive SHAP) and which push it DOWN (negative SHAP).

---

## 10. recommendation_engine.py — Skill Gaps & Roadmaps

**Purpose:** Identify a student's weakest and strongest areas, then generate a personalised improvement plan.

### Skill Categories

The engine evaluates 6 categories:
- **Academic**: Based on CGPA and AcademicScore
- **Technical**: Based on CodingIndex and TechnicalScore
- **Communication**: Based on CommunicationScore
- **Experience**: Based on ExperienceScore (internships + certifications)
- **Projects**: Based on ProjectScore
- **Leadership**: Based on Soft_Skills_Rating

#### `analyse_skill_gaps(norm_data) → dict`
Calculates percentage scores for each category. Categories below 60% are weaknesses.

#### `generate_roadmap(weaknesses, strengths) → list`
Creates a 4-week plan focusing on the student's weakest areas first, then intermediate and advanced topics.

---

## 11. company_matching.py — Mapping to Companies

**Purpose:** Map placement probability to realistic Indian company tiers.

### Company Tiers

| Tier | Threshold | Example Companies |
|------|-----------|------------------|
| Dream Companies | >80% prob | Google, Microsoft, Amazon, Meta |
| Aspirational | >60% | Flipkart, Uber, Goldman Sachs, Adobe |
| Strong Companies | >40% | TCS Digital, Infosys PP, Mindtree |
| Mass Recruiters | >20% | TCS, Infosys, Accenture, Wipro |

#### `match_companies(probability, norm_data) → dict`
Returns eligible tiers, stretch targets, and companies that need improvement to reach.

---

## 12. resume_scoring.py — ATS Resume Analysis

**Purpose:** Score resume text against ATS (Applicant Tracking System) keyword matching.

#### `score_resume(text) → dict`
1. Counts technical skills found (Python, Java, Docker, etc.)
2. Counts action verbs (developed, implemented, etc.)
3. Checks for essential sections (Education, Experience, Skills, etc.)
4. Checks formatting keywords (email, LinkedIn, GitHub, etc.)
5. Calculates overall ATS score (0–100%)
6. Provides improvement suggestions

---

## 13. interview_scoring.py — Interview Evaluation

**Purpose:** Evaluate mock interview performance across 4 dimensions.

#### `evaluate_interview(technical_score, communication_score, confidence_score, question_accuracy) → dict`
Weighted scoring:
- Technical: 35%
- Communication: 25%
- Confidence: 20%
- Question Accuracy: 20%

Returns overall score, readiness level, weak areas, and improvement suggestions.

---

## 14. report_generator.py — Building the Final Report

**Purpose:** Assemble all analysis results into a beautiful, formatted text report.

#### `generate_report(probability, cluster_name, skill_analysis, roadmap, company_matching, explanation, ...) → str`
Builds a multi-section report with:
- Box-drawing Unicode borders
- Emoji icons for visual appeal
- All prediction results, explanations, and recommendations
- A standard disclaimer

#### `save_report(report, output_dir)`
Writes the report to `outputs/placement_report.txt`.

#### `classify_readiness(probability) → str`
Maps probability to readiness level:
- ≥ 85%: "Exceptional"
- ≥ 70%: "Highly Competitive"
- ≥ 55%: "Competitive"
- ≥ 40%: "Improving"
- < 40%: "Beginner"

---

## 15. llm_resume_classifier.py — DistilBERT Fine-Tuning

**Purpose:** Fine-tune a pre-trained DistilBERT transformer model to classify resumes as Placed/Not Placed.

#### `generate_resume_text(row) → str`
Converts a CSV row into a natural language paragraph describing the student's profile.

#### `train_llm_model(train_csv, model_dir)`
1. Loads `train.csv`
2. Generates resume paragraphs for each student
3. Tokenises text using `DistilBertTokenizer`
4. Creates a PyTorch Dataset
5. Fine-tunes `DistilBertForSequenceClassification` for 3 epochs
6. Saves model + tokenizer to `models/llm_resume_model/`

#### `predict_resume_llm(text, model, tokenizer, device) → dict`
Classifies a single resume text and returns `{"label": "Placed", "confidence": 0.92}`.

---

## 16. Data Flow Diagram

```
User input (interactive) OR CSV file (batch)
        │
        ▼
┌───────────────────┐
│ Raw Student Data   │  {"CGPA": 8.5, "Coding_Skills": 8, ...}
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineering│  Creates AcademicScore, TechnicalScore, etc.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Encode + Scale     │  LabelEncoder (categoricals) + StandardScaler
└────────┬──────────┘
         │
    ┌────┼────┬──────────┐
    │    │    │           │
    ▼    ▼    ▼           ▼
 Classifier  KMeans  SHAP     Skill Analysis
 predict()   assign   explain  identify gaps
    │    │    │           │
    ▼    ▼    ▼           ▼
 Probability Cluster  Explanation  Roadmap
    │    │    │           │
    └────┼────┴───────────┘
         │
         ▼
┌───────────────────┐
│ Report Generator   │  Assembles everything into final report
└────────┬──────────┘
         │
         ▼
   placement_report.txt
```
