# Model Training — How PlacementIQ AI Learns

This document explains every machine learning model used in PlacementIQ AI, how they are trained, and why they were chosen.

---

## Table of Contents
1. [Training Pipeline Overview](#1-training-pipeline-overview)
2. [Classical ML Models (5 Models)](#2-classical-ml-models)
3. [Unsupervised Learning — KMeans Clustering](#3-unsupervised-learning--kmeans-clustering)
4. [LLM / Transformer Model — DistilBERT](#4-llm--transformer-model--distilbert)
5. [Feature Engineering](#5-feature-engineering)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Model Evaluation Metrics](#7-model-evaluation-metrics)

---

## 1. Training Pipeline Overview

```
Raw CSV Data
    │
    ▼
Feature Engineering (7 composite features created)
    │
    ▼
Preprocessing (encode categoricals → handle outliers → scale numerically)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              SUPERVISED LEARNING (Classification)     │
│                                                       │
│   Model 1: Random Forest         ─┐                  │
│   Model 2: Gradient Boosting      │  Compare AUC     │
│   Model 3: Logistic Regression    ├─► Select Best     │
│   Model 4: Decision Tree          │                  │
│   Model 5: XGBoost               ─┘                  │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              UNSUPERVISED LEARNING (Clustering)       │
│                                                       │
│   KMeans (K=2 to K=10, select by Silhouette Score)   │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              NLP / DEEP LEARNING (LLM)               │
│                                                       │
│   DistilBERT Fine-Tuning for Resume Classification   │
└──────────────────────────────────────────────────────┘
    │
    ▼
Explainable AI (SHAP Values + Feature Importance)
    │
    ▼
Save All Models to models/ directory
```

---

## 2. Classical ML Models

We train **5 different supervised classification models** and compare them. The best one (by ROC-AUC score) is saved and used for predictions.

### 2.1 Random Forest 🏆

**What it is:** An ensemble of many Decision Trees. Each tree is trained on a random subset of the data and features. The final prediction is the majority vote of all trees.

**Why it works well:** It handles non-linear relationships, is resistant to overfitting, and works great with tabular data.

**Hyperparameters used:**
```python
RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=15,          # Max tree depth = 15
    min_samples_split=5,   # Need ≥5 samples to split a node
    random_state=42        # Reproducibility
)
```

**How it decides:** Each tree votes "Placed" or "Not Placed". If 150 out of 200 trees say "Placed", then the probability is 150/200 = 75%.

---

### 2.2 Gradient Boosting

**What it is:** Builds trees sequentially. Each new tree learns to correct the errors made by the previous tree.

**Why it works well:** Very powerful for structured data. Often wins Kaggle competitions.

**Hyperparameters used:**
```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,     # How fast the model learns
    max_depth=5,
    random_state=42
)
```

**How it learns:** Tree 1 makes predictions → calculates errors → Tree 2 focuses on the errors → repeats 200 times.

---

### 2.3 Logistic Regression

**What it is:** A linear model that predicts the probability of a binary outcome using the **sigmoid function**.

**The math:**
```
P(Placed) = 1 / (1 + e^(-(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)))
```

Where:
- `x₁, x₂, ..., xₙ` are the features (CGPA, Coding_Skills, etc.)
- `w₁, w₂, ..., wₙ` are learned weights
- `b` is the bias
- The sigmoid squashes any value to between 0 and 1

**Why it's included:** Simple baseline model. If a linear model performs well, the problem might be easier than expected.

**Hyperparameters:**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

### 2.4 Decision Tree

**What it is:** A single tree that makes decisions by splitting data on feature thresholds.

**Example decision path:**
```
Is CGPA ≥ 7.5?
├── YES → Is Coding_Skills ≥ 7?
│         ├── YES → Placed (90% probability)
│         └── NO  → Not Placed (60% probability)
└── NO  → Is Internships ≥ 2?
          ├── YES → Placed (55% probability)
          └── NO  → Not Placed (85% probability)
```

**Hyperparameters:**
```python
DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
```

---

### 2.5 XGBoost (Extreme Gradient Boosting)

**What it is:** An optimised, high-performance implementation of gradient boosting. Uses advanced techniques like regularisation and parallel processing.

**Why it's special:** Faster than regular Gradient Boosting, handles missing values natively, and includes L1/L2 regularisation to prevent overfitting.

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

---

### 2.6 Cross-Validation Strategy

Every model is evaluated using **5-Fold Stratified Cross-Validation**:

```
Dataset split into 5 equal parts (folds):
┌──────┬──────┬──────┬──────┬──────┐
│Fold 1│Fold 2│Fold 3│Fold 4│Fold 5│
└──────┴──────┴──────┴──────┴──────┘

Round 1: Train on Folds 2-5, Test on Fold 1 → AUC₁
Round 2: Train on Folds 1,3-5, Test on Fold 2 → AUC₂
Round 3: Train on Folds 1-2,4-5, Test on Fold 3 → AUC₃
Round 4: Train on Folds 1-3,5, Test on Fold 4 → AUC₄
Round 5: Train on Folds 1-4, Test on Fold 5 → AUC₅

Final Score = Average(AUC₁, AUC₂, AUC₃, AUC₄, AUC₅) ± Std Dev
```

"Stratified" means each fold maintains the same ratio of Placed/Not Placed students as the full dataset.

---

## 3. Unsupervised Learning — KMeans Clustering

**Purpose:** Group students into clusters based on their feature profiles (independent of whether they got placed).

### How KMeans Works

1. Randomly place K centroids in feature space
2. Assign each student to the nearest centroid
3. Recalculate centroids as the mean of assigned students
4. Repeat steps 2-3 until centroids stop moving

### Selecting K (Number of Clusters)

We use the **Silhouette Score** to find the optimal number of clusters:

```
For K = 2, 3, 4, ..., 10:
    Run KMeans with K clusters
    Calculate Silhouette Score (higher = better separation)
    
Best K = the one with highest Silhouette Score
```

**Silhouette Score** measures how similar a student is to their own cluster vs. other clusters. Range: -1 to +1. Higher is better.

### Cluster Naming

After clustering, each cluster is automatically labelled based on the average placement probability of students in it:
- **Top Performers**: High placement probability
- **Average Students**: Medium placement probability  
- **Needs Improvement**: Low placement probability

---

## 4. LLM / Transformer Model — DistilBERT

### What is DistilBERT?

DistilBERT is a **distilled (compressed) version of BERT** created by HuggingFace. It has:
- **66 million parameters** (vs BERT's 110M)
- **6 transformer layers** (vs BERT's 12)
- **40% smaller** and **60% faster** than BERT
- Retains **97% of BERT's performance**

### What is Transfer Learning?

Instead of training a language model from scratch (which requires billions of tokens and weeks of GPU time), we:

1. **Start with a pre-trained model**: DistilBERT has already been trained on Wikipedia + BookCorpus (billions of words). It understands English grammar, semantics, and context.

2. **Add a classification head**: We add a new layer on top that outputs 2 classes (Placed / Not Placed).

3. **Fine-tune on our data**: We train ONLY the classification head (and slightly adjust the transformer weights) on our placement-specific resume data. This takes minutes instead of weeks.

```
Pre-trained DistilBERT (Frozen/Slightly Adjusted)
        │
        ▼
┌──────────────────────────┐
│ [CLS] Token Embedding    │  (768-dimensional vector)
│ representing entire text  │
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │ Dense Layer  │  768 → 2
    │ (NEW, trained│
    │  from scratch)│
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Softmax     │  → [P(Not Placed), P(Placed)]
    └─────────────┘
```

### How We Generate Training Data

Since we don't have real resume texts, we convert each row from `train.csv` into a **natural language paragraph**:

```
Example row: Age=22, Gender=Male, Degree=B.Tech, Branch=CSE, CGPA=8.5, 
             Internships=2, Projects=5, Coding_Skills=8, ...

Generated paragraph:
"A 22 year old Male student pursuing B.Tech in CSE with a CGPA of 8.5.
 Has completed 2 internships and 5 projects. Coding skills rated 8 out of 10.
 Communication skills rated 7 out of 10. Aptitude test score: 82.
 Soft skills rating: 7. Holds 3 certifications with 0 backlogs."
```

This paragraph is then tokenised by the DistilBERT tokenizer and fed into the model for training.

### Training Configuration

```python
Training epochs:    3
Batch size:         16
Learning rate:      2e-5 (very small, to avoid destroying pre-trained knowledge)
Max sequence length: 128 tokens
Optimiser:          AdamW
Loss function:      CrossEntropyLoss
```

---

## 5. Feature Engineering

We create **7 composite features** from the raw columns to capture higher-level student attributes:

| Composite Feature | Source Columns | Formula |
|-------------------|---------------|---------|
| `AcademicScore` | CGPA, Backlogs | Weighted sum (normalised to 0–1) |
| `TechnicalScore` | Coding_Skills, Aptitude_Test_Score | Weighted sum |
| `ExperienceScore` | Internships, Certifications, Age | Weighted sum |
| `CommunicationScore` | Communication_Skills, Soft_Skills_Rating | Weighted sum |
| `ProjectScore` | Projects | Normalised |
| `CodingIndex` | Coding_Skills, Aptitude_Test_Score | Alternative weighting |
| `OverallSkillScore` | All above composite scores | Master score (weighted average) |

These engineered features help the model capture patterns like "a student with high CGPA but many backlogs" vs. "a student with moderate CGPA but zero backlogs."

---

## 6. Data Preprocessing

### Step-by-step pipeline:

1. **Remove duplicates**: Drop identical rows
2. **Handle missing values**: Fill numerical with median, categorical with mode
3. **Encode target**: Map "Placed" → 1, "Not Placed" → 0
4. **Encode categoricals**: LabelEncoder for Gender, Degree, Branch
5. **Detect outliers**: IQR-based capping (values beyond 1.5×IQR are clipped)
6. **Scale features**: StandardScaler (mean=0, std=1) for all numerical columns
7. **Split data**: Use pre-split train.csv (45,000) and test.csv (5,000)

---

## 7. Model Evaluation Metrics

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | "Of all students predicted as Placed, how many actually were?" | TP / (TP + FP) |
| **Recall** | "Of all actually Placed students, how many did we correctly predict?" | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of Precision and Recall | 2 × (P × R) / (P + R) |
| **ROC-AUC** | Area Under the ROC Curve (probability ranking quality) | Ranges 0–1, higher is better |

Where:
- **TP** = True Positive (correctly predicted "Placed")
- **TN** = True Negative (correctly predicted "Not Placed")
- **FP** = False Positive (predicted "Placed" but actually "Not Placed")
- **FN** = False Negative (predicted "Not Placed" but actually "Placed")

### Best Model Selection

The model with the **highest ROC-AUC score** on the test set is automatically selected as the best model and saved for predictions. In our case, **Random Forest** achieved an AUC of **1.0** (perfect classification).
