# 📘 Python Libraries & Data Pre-processing — Complete Guide
### PlacementIQ AI Project — Comprehensive Reference

> **This document covers two major topics with every sub-section explained in detail and mapped to the actual PlacementIQ AI codebase.**

---

## Table of Contents

### Part A — Introduction to Google Colab & Essential Python Libraries
1. [Google Colab — Overview & Usage](#1-google-colab--overview--usage)
2. [NumPy — Numerical Computing](#2-numpy--numerical-computing)
3. [Pandas — Data Manipulation & Analysis](#3-pandas--data-manipulation--analysis)
4. [Matplotlib — Data Visualization](#4-matplotlib--data-visualization)
5. [Scikit-Learn — Machine Learning](#5-scikit-learn--machine-learning)

### Part B — Data Pre-processing & Feature Engineering
6. [Data Collection](#6-data-collection)
7. [Data Cleaning](#7-data-cleaning)
8. [Handling Missing Values](#8-handling-missing-values)
9. [Data Integration](#9-data-integration)
10. [Data Transformation](#10-data-transformation)
11. [Data Reduction](#11-data-reduction)
12. [Feature Scaling — Normalization & Standardization](#12-feature-scaling--normalization--standardization)
13. [Feature Engineering](#13-feature-engineering)
14. [Feature Selection](#14-feature-selection)
15. [Feature Extraction](#15-feature-extraction)
16. [Preparation of Training & Testing Datasets](#16-preparation-of-training--testing-datasets)

---

# Part A — Introduction to Google Colab & Essential Python Libraries

---

## 1. Google Colab — Overview & Usage

### 1.1 What is Google Colab?

Google Colab (Colaboratory) is a **free, cloud-based Jupyter Notebook environment** provided by Google. It allows you to write and execute Python code directly in a web browser without any local setup.

**Key features:**
- **Zero setup required** — runs entirely in the cloud
- **Free GPU/TPU access** — for accelerating ML training
- **Pre-installed libraries** — NumPy, Pandas, Scikit-Learn, Matplotlib, TensorFlow, and PyTorch come pre-installed
- **Google Drive integration** — seamless file storage and sharing
- **Collaboration** — multiple users can edit the same notebook in real-time

### 1.2 How Our Project Uses Google Colab

Our PlacementIQ AI project includes **two Colab-compatible notebooks**:

| Notebook | Purpose |
|----------|---------|
| `colab_preprocessing.ipynb` | Complete data preprocessing pipeline — loading data, finding missing values, visualizations, encoding, scaling, and train-test split |
| `accuracy_comparison.ipynb` | Comparing model accuracy with original data vs. engineered features |

### 1.3 Setting Up Google Colab

**Step 1 — Open Google Colab:**
- Navigate to [https://colab.research.google.com](https://colab.research.google.com)
- Sign in with your Google account

**Step 2 — Upload or Mount Data:**
```python
# Option A: Upload dataset directly from Google Drive
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/dataset/train.csv')

# Option B: Upload from local machine
from google.colab import files
uploaded = files.upload()
```

**Step 3 — Install additional libraries (if needed):**
```python
!pip install xgboost shap
```

### 1.4 Google Colab vs Local Execution

Our project supports **both** environments. The notebook `colab_preprocessing.ipynb` was originally designed for Colab but has been modified to also run locally:

```python
# In Colab:
df = pd.read_csv('/content/drive/MyDrive/dataset/train.csv')

# Locally (as used in our project):
df = pd.read_csv('dataset/train.csv')
```

> **In our project:** The `colab_preprocessing.ipynb` notebook handles the complete preprocessing pipeline step by step, with visualizations at each stage.

---

## 2. NumPy — Numerical Computing

### 2.1 What is NumPy?

NumPy (Numerical Python) is the **foundational library for numerical computing** in Python. It provides:
- **N-dimensional arrays** (`ndarray`) — efficient multi-dimensional data structures
- **Mathematical operations** — linear algebra, statistics, random number generation
- **Broadcasting** — performing operations on arrays of different shapes
- **Vectorized computation** — fast element-wise operations without explicit loops

### 2.2 Why NumPy is Essential

| Feature | Benefit |
|---------|---------|
| `ndarray` | 50× faster than Python lists for numerical operations |
| Vectorization | Eliminates slow Python loops |
| Broadcasting | Automatically handles shape-mismatched operations |
| Memory efficiency | Contiguous memory layout for cache-friendly access |

### 2.3 How Our Project Uses NumPy

NumPy is imported and used across **every module** in our project:

**a) In `data_loader.py` — Correlation Heatmap Masking:**
```python
import numpy as np

# Create an upper-triangle mask for the correlation heatmap
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
```
Here, `np.triu()` creates an upper-triangular matrix to avoid duplicate correlations in the heatmap.

**b) In `preprocessing.py` — Column Type Detection:**
```python
numerical = df.select_dtypes(include=[np.number]).columns.tolist()
```
Uses `np.number` as a dtype selector to automatically find all numeric columns.

**c) In `feature_engineering.py` — Weighted Score Calculation:**
```python
score = sum(w * normalised[c] for c, w in zip(avail_cols, avail_weights))
```
Internally, Pandas Series operations use NumPy arrays for fast element-wise weighted sums.

**d) In `train_cluster.py` — Optimal K Selection:**
```python
best_k = k_range.start + int(np.argmax(silhouettes))
```
Uses `np.argmax()` to find the K value with the highest silhouette score.

**e) In `evaluate_classifier.py` — Predictions & Probabilities:**
```python
y_proba = model.predict_proba(X_test)[:, 1]
```
NumPy array slicing extracts the probability of the positive class.

### 2.4 Key NumPy Operations Used

| Operation | Usage in Project | Location |
|-----------|-----------------|----------|
| `np.triu()` | Correlation mask | `data_loader.py` |
| `np.number` | Numeric dtype detection | `data_loader.py` |
| `np.argmax()` | Best cluster K selection | `train_cluster.py` |
| `np.argsort()` | Ranking clusters by strength | `train_cluster.py` |
| `np.array().flatten()` | Subplot axis manipulation | `data_loader.py` |

---

## 3. Pandas — Data Manipulation & Analysis

### 3.1 What is Pandas?

Pandas is the **primary data manipulation library** in Python. It provides:
- **DataFrame** — a 2D labeled table (like Excel or SQL table)
- **Series** — a 1D labeled array (a single column)
- **Data I/O** — reading/writing CSV, Excel, JSON, SQL, etc.
- **Data wrangling** — filtering, grouping, merging, pivoting, reshaping

### 3.2 Why Pandas is Essential

| Feature | Benefit |
|---------|---------|
| `pd.read_csv()` | Load data from any file format in one line |
| `df.describe()` | Instant statistical summary |
| `df.isnull()` | Find all missing values |
| `df.fillna()` | Impute missing values |
| `df.drop_duplicates()` | Remove duplicate rows |
| `df.select_dtypes()` | Separate numerical/categorical columns |

### 3.3 How Our Project Uses Pandas

**a) Loading Data (`data_loader.py`):**
```python
import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset from %s — shape %s", filepath, df.shape)
    return df
```

**b) Missing Value Report (`data_loader.py`):**
```python
def missing_value_report(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing": missing, "Percent": pct})
    report = report[report["Missing"] > 0].sort_values("Percent", ascending=False)
    return report
```

**c) Feature Summary — Descriptive Statistics (`data_loader.py`):**
```python
def feature_summary(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    summary = df.describe(include="all").T
    summary.to_csv(summary_path)
    return summary
```
Uses `df.describe(include="all")` to generate count, mean, std, min, 25%, 50%, 75%, max for every feature.

**d) Column Auto-Detection (`data_loader.py`):**
```python
categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = df.select_dtypes(include=[np.number]).columns.tolist()
```

**e) Data Concatenation for Consistent Encoding (`preprocessing.py`):**
```python
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
```
Concatenates train and test for consistent encoding and scaling, then splits back.

**f) Model Comparison Table (`evaluate_classifier.py`):**
```python
comparison_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
```

### 3.4 Key Pandas Operations Used

| Operation | Purpose | Location |
|-----------|---------|----------|
| `pd.read_csv()` | Load CSV datasets | `data_loader.py` |
| `df.describe()` | Statistical summary | `data_loader.py` |
| `df.isnull().sum()` | Missing value detection | `data_loader.py` |
| `df.fillna()` | Missing value imputation | `preprocessing.py` |
| `df.drop_duplicates()` | Duplicate removal | `preprocessing.py` |
| `df.select_dtypes()` | Column type filtering | `data_loader.py` |
| `pd.concat()` | Train-test combination | `preprocessing.py` |
| `df.to_csv()` | Save cleaned data | `preprocessing.py` |
| `df.value_counts()` | Class distribution | `data_loader.py` |
| `df.clip()` | Outlier capping | `preprocessing.py` |
| `df.quantile()` | IQR calculation | `preprocessing.py` |

---

## 4. Matplotlib — Data Visualization

### 4.1 What is Matplotlib?

Matplotlib is the **most widely used plotting library** in Python. It provides:
- **Line plots, bar charts, histograms, scatter plots, pie charts**
- **Heatmaps, boxplots** (via Seaborn integration)
- **Subplots** — multiple plots in one figure
- **Customization** — colors, fonts, labels, legends, annotations

### 4.2 Seaborn — Statistical Visualization

Our project also uses **Seaborn**, which is built on top of Matplotlib and provides:
- More aesthetically pleasing default styles
- Statistical plot types (boxplots, violin plots, heatmaps)
- Automatic color palettes

### 4.3 How Our Project Uses Matplotlib

**a) Backend Configuration:**
```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves plots to files
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
```

**b) Correlation Heatmap (`data_loader.py`):**
```python
def plot_correlation_heatmap(df, output_dir):
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
```

**c) Class Distribution Bar Chart (`data_loader.py`):**
```python
def plot_class_distribution(df, target_col, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    counts.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    fig.savefig(path)
```

**d) Feature Histograms (`data_loader.py`):**
```python
def plot_histograms(df, numerical_cols, output_dir):
    n_plot_cols = 3
    n_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
    fig, axes = plt.subplots(n_rows, n_plot_cols, ...)
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=25, color="#3498db",
                       edgecolor="white", alpha=0.85)
```

**e) Boxplots for Outlier Detection (`data_loader.py`):**
```python
def plot_boxplots(df, numerical_cols, output_dir):
    for idx, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[idx], color="#9b59b6", width=0.4)
```

**f) Elbow & Silhouette Plot (`train_cluster.py`):**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_info["k_range"], k_info["inertias"], "bo-", linewidth=2)
ax2.plot(k_info["k_range"], k_info["silhouettes"], "go-", linewidth=2)
```

**g) ROC Curve Comparison (`evaluate_classifier.py`):**
```python
def plot_roc_curves(all_models, X_test, y_test, output_dir):
    for (name, model), color in zip(all_models.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
```

**h) Confusion Matrix (`evaluate_classifier.py`):**
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"])
```

### 4.4 All Visualizations Generated

| Plot Type | File Saved | Module |
|-----------|-----------|--------|
| Correlation Heatmap | `correlation_heatmap.png` | `data_loader.py` |
| Class Distribution | `class_distribution.png` | `data_loader.py` |
| Feature Histograms | `histograms.png` | `data_loader.py` |
| Feature Boxplots | `boxplots.png` | `data_loader.py` |
| Missing Values Heatmap | (in notebook) | `colab_preprocessing.ipynb` |
| Elbow + Silhouette | `cluster_elbow_silhouette.png` | `train_cluster.py` |
| Cluster Distribution | `cluster_distribution.png` | `train_cluster.py` |
| ROC Curves | `roc_curves_comparison.png` | `evaluate_classifier.py` |
| Confusion Matrix | `confusion_matrix_*.png` | `evaluate_classifier.py` |
| Model Comparison | `model_comparison.png` | `evaluate_classifier.py` |
| SHAP Summary Plot | `shap_summary.png` | `explain_prediction.py` |
| Feature Importance | `feature_importance.png` | `explain_prediction.py` |

---

## 5. Scikit-Learn — Machine Learning

### 5.1 What is Scikit-Learn?

Scikit-Learn (sklearn) is the **most popular machine learning library** in Python. It provides:
- **Preprocessing** — encoding, scaling, imputation
- **Model selection** — train-test split, cross-validation, hyperparameter tuning
- **Classification** — Random Forest, Logistic Regression, Decision Trees, Gradient Boosting
- **Clustering** — KMeans, DBSCAN
- **Evaluation** — accuracy, precision, recall, F1, ROC AUC, confusion matrix

### 5.2 How Our Project Uses Scikit-Learn

**a) Preprocessing (`preprocessing.py`):**
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# LabelEncoder — converts categorical text to numbers
le = LabelEncoder()
df[col] = le.fit_transform(df[col].astype(str))

# StandardScaler — standardizes features (zero mean, unit variance)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

**b) Feature Engineering (`feature_engineering.py`):**
```python
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler — normalizes features to [0, 1]
scaler = MinMaxScaler()
df[existing] = scaler.fit_transform(df[existing])
```

**c) Classification Models (`train_classifier.py`):**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Five models trained and compared:
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=8),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6),
}
```

**d) Clustering (`train_cluster.py`):**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
labels = km.fit_predict(X)
sil = silhouette_score(X, labels)
```

**e) Evaluation (`evaluate_classifier.py`):**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
```

### 5.3 Complete Scikit-Learn Usage Map

| sklearn Module | Class/Function | Purpose | Our File |
|---------------|---------------|---------|----------|
| `preprocessing` | `LabelEncoder` | Encode categorical columns | `preprocessing.py` |
| `preprocessing` | `StandardScaler` | Standardize numerical features | `preprocessing.py` |
| `preprocessing` | `MinMaxScaler` | Normalize to [0, 1] | `feature_engineering.py` |
| `model_selection` | `train_test_split` | Split data into train/test | `preprocessing.py` |
| `model_selection` | `cross_val_score` | K-fold cross-validation | `train_classifier.py` |
| `ensemble` | `RandomForestClassifier` | Ensemble tree classifier | `train_classifier.py` |
| `ensemble` | `GradientBoostingClassifier` | Boosted tree classifier | `train_classifier.py` |
| `linear_model` | `LogisticRegression` | Linear classifier | `train_classifier.py` |
| `tree` | `DecisionTreeClassifier` | Single decision tree | `train_classifier.py` |
| `cluster` | `KMeans` | Student clustering | `train_cluster.py` |
| `metrics` | `accuracy_score` | Classification accuracy | `evaluate_classifier.py` |
| `metrics` | `precision_score` | Positive predictive value | `evaluate_classifier.py` |
| `metrics` | `recall_score` | True positive rate | `evaluate_classifier.py` |
| `metrics` | `f1_score` | Harmonic mean of P & R | `evaluate_classifier.py` |
| `metrics` | `roc_auc_score` | Area under ROC curve | `evaluate_classifier.py` |
| `metrics` | `roc_curve` | ROC plot data | `evaluate_classifier.py` |
| `metrics` | `confusion_matrix` | TP/TN/FP/FN counts | `evaluate_classifier.py` |
| `metrics` | `classification_report` | Full classification report | `evaluate_classifier.py` |
| `metrics` | `silhouette_score` | Cluster quality metric | `train_cluster.py` |

---

# Part B — Data Pre-processing & Feature Engineering

---

## 6. Data Collection

### 6.1 What is Data Collection?

Data collection is the **first step** in any ML project. It involves gathering raw data from various sources. The quality and quantity of collected data directly impacts model performance.

### 6.2 Data Sources

Common data sources include:
- **CSV/Excel files** — tabular datasets
- **Databases** — SQL, NoSQL (MongoDB)
- **APIs** — web services providing structured data
- **Web scraping** — extracting data from websites
- **Surveys/Forms** — manually collected data

### 6.3 How Our Project Collects Data

Our project uses a **pre-collected dataset** of **45,000 student records** in CSV format, split into training and testing sets:

```
dataset/
├── train.csv    (2.5 MB — ~40,500 training records)
└── test.csv     (281 KB — ~4,500 testing records)
```

**Dataset columns (15 features):**

| Column | Type | Description |
|--------|------|-------------|
| `Student_ID` | Integer | Unique identifier (not a feature) |
| `Age` | Integer | Student age |
| `Gender` | Categorical | Male / Female |
| `Degree` | Categorical | B.Tech, BCA, MCA, etc. |
| `Branch` | Categorical | CSE, ECE, ME, etc. |
| `CGPA` | Float | Cumulative Grade Point Average |
| `Internships` | Integer | Number of internships completed |
| `Projects` | Integer | Number of projects completed |
| `Coding_Skills` | Integer | Coding skill rating (1–10) |
| `Communication_Skills` | Integer | Communication rating (1–10) |
| `Aptitude_Test_Score` | Integer | Aptitude score (0–100) |
| `Soft_Skills_Rating` | Integer | Soft skills rating (1–10) |
| `Certifications` | Integer | Number of certifications |
| `Backlogs` | Integer | Number of academic backlogs |
| `Placement_Status` | Categorical | **Target** — "Placed" / "Not Placed" |

**Loading code (`data_loader.py`):**
```python
def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset from %s — shape %s", filepath, df.shape)
    return df
```

---

## 7. Data Cleaning

### 7.1 What is Data Cleaning?

Data cleaning (data cleansing) is the process of **detecting and correcting** errors, inconsistencies, and inaccuracies in the data. This includes:
- Removing duplicate records
- Fixing structural errors (typos, inconsistent naming)
- Removing irrelevant features (IDs, timestamps)
- Handling data type mismatches

### 7.2 Why Data Cleaning Matters

| Problem | Impact | Solution |
|---------|--------|----------|
| Duplicate rows | Biases model toward repeated patterns | `drop_duplicates()` |
| Irrelevant columns | Adds noise, increases training time | Remove `Student_ID` |
| Inconsistent values | Creates false categories | Standardize text |
| Wrong data types | Causes computation errors | Type casting |

### 7.3 How Our Project Cleans Data

**a) Duplicate Removal (`preprocessing.py`):**
```python
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (before=%d, after=%d)",
                removed, before, len(df))
    return df
```

**b) Irrelevant Column Removal (`data_loader.py`):**
```python
# Student_ID is automatically excluded from feature lists
if "Student_ID" in numerical:
    numerical.remove("Student_ID")
if "Student_ID" in categorical:
    categorical.remove("Student_ID")
```

**c) Outlier Detection & Capping (`preprocessing.py`):**
```python
def detect_outliers(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df
```
Uses the **IQR (Interquartile Range)** method to detect outliers and cap them to the [Q1 − 1.5·IQR, Q3 + 1.5·IQR] range.

---

## 8. Handling Missing Values

### 8.1 What are Missing Values?

Missing values are **absent entries** in the dataset. They can occur due to:
- Data collection errors
- Survey non-responses
- System failures
- Intentionally left blank

### 8.2 Types of Missing Data

| Type | Description | Example |
|------|-------------|---------|
| **MCAR** (Missing Completely at Random) | No pattern in missingness | Random sensor failure |
| **MAR** (Missing at Random) | Depends on observed variables | Higher earners skip income question |
| **MNAR** (Missing Not at Random) | Depends on the missing value itself | Very sick patients skip health survey |

### 8.3 Strategies for Handling Missing Values

| Strategy | When to Use | Our Implementation |
|----------|------------|-------------------|
| **Mean/Median imputation** | Numerical columns with few missing values | ✅ Median for numerical |
| **Mode imputation** | Categorical columns | ✅ Mode for categorical |
| **Forward/Backward fill** | Time-series data | ❌ Not applicable |
| **Deletion** | Very few missing rows | ❌ Not used |
| **Model-based (KNN, MICE)** | Complex patterns | ❌ Not needed |

### 8.4 How Our Project Handles Missing Values

**Detection (`data_loader.py`):**
```python
def missing_value_report(df, output_dir):
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing": missing, "Percent": pct})
    report = report[report["Missing"] > 0].sort_values("Percent", ascending=False)
    return report
```

**Imputation (`preprocessing.py`):**
```python
def handle_missing_values(df, numerical_cols, categorical_cols):
    # Numerical → fill with MEDIAN (robust to outliers)
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical → fill with MODE (most frequent value)
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    remaining = df.isnull().sum().sum()
    logger.info("Remaining missing values: %d", remaining)
    return df
```

**In our Colab notebook (`colab_preprocessing.ipynb`):**
```python
# Step 2: Find Null (Missing) Values
print("Total missing values in entire dataset:", df.isnull().sum().sum())
print("Missing values per column:\n", df.isnull().sum())
missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
print(f"Overall dataset missing percentage: {missing_percentage:.2f}%")
```

> **Result in our project:** The dataset has **0 missing values** (0.00%), but the imputation code is still included for robustness.

---

## 9. Data Integration

### 9.1 What is Data Integration?

Data integration is the process of **combining data from multiple sources** into a single, unified dataset. This is critical when:
- Data comes from different databases or files
- Training and testing data are in separate files
- External data needs to be merged (e.g., company data)

### 9.2 Challenges in Data Integration

| Challenge | Description |
|-----------|-------------|
| **Schema conflicts** | Different column names for the same attribute |
| **Data redundancy** | Same information duplicated across sources |
| **Entity identification** | Matching records across datasets |
| **Value conflicts** | Different units or scales for the same attribute |

### 9.3 How Our Project Integrates Data

**Combining Train and Test Datasets (`preprocessing.py`):**
```python
# Store sizes to split back later
n_train = len(df_train)

# Concatenate for consistent encoding and scaling
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
```

This is done so that:
- **LabelEncoder** learns all categories from both sets (avoids unseen labels in test)
- **StandardScaler** is fitted on the combined distribution
- **Outlier detection** considers the full data range

After preprocessing, the data is split back:
```python
X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test = X.iloc[n_train:]
y_test = y.iloc[n_train:]
```

---

## 10. Data Transformation

### 10.1 What is Data Transformation?

Data transformation converts data into a **format suitable for analysis and modeling**. This includes:
- **Encoding** — converting categories to numbers
- **Scaling** — adjusting numerical ranges
- **Normalization** — mapping to [0, 1]
- **Log/Power transforms** — reducing skewness
- **Target encoding** — converting target labels

### 10.2 Types of Transformations

| Type | Method | Purpose |
|------|--------|---------|
| **Label Encoding** | `LabelEncoder` | Convert text categories to integers |
| **One-Hot Encoding** | `pd.get_dummies()` | Create binary columns per category |
| **Ordinal Encoding** | Manual mapping | Preserve natural order |
| **Target Mapping** | Custom dictionary | Convert target labels |

### 10.3 How Our Project Transforms Data

**a) Categorical Encoding (`preprocessing.py`):**
```python
def encode_categoricals(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded %s — classes: %s", col, list(le.classes_))
    return df, encoders
```

Example transformation:
| Before | After |
|--------|-------|
| `Gender: Male` | `Gender: 1` |
| `Gender: Female` | `Gender: 0` |
| `Degree: B.Tech` | `Degree: 0` |
| `Degree: BCA` | `Degree: 1` |
| `Degree: MCA` | `Degree: 2` |

**b) Target Column Encoding (`preprocessing.py`):**
```python
# Special handling for "Placed"/"Not Placed" target
if df[target_col].dtype == object:
    unique_vals = set(df[target_col].dropna().unique())
    if unique_vals == {"Placed", "Not Placed"}:
        df[target_col] = df[target_col].map({"Placed": 1, "Not Placed": 0})
```

**c) Saving Transformers for Inference (`preprocessing.py`):**
```python
def save_transformers(scaler, encoders, model_dir):
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "encoders.joblib"))
```
All fitted encoders and scalers are saved via Joblib, so the **same transformations** can be applied during prediction/inference.

---

## 11. Data Reduction

### 11.1 What is Data Reduction?

Data reduction aims to **reduce the volume** of data while maintaining its integrity. Techniques include:
- **Dimensionality reduction** — reducing number of features (PCA, t-SNE)
- **Numerosity reduction** — reducing number of rows (sampling)
- **Data compression** — encoding data more compactly
- **Feature removal** — dropping irrelevant/redundant features

### 11.2 How Our Project Reduces Data

**a) Irrelevant Feature Removal:**
```python
# Student_ID is removed as it carries no predictive value
feature_cols = [c for c in df_scaled.columns
                if c != target_col and c != "Student_ID"]
```

**b) Duplicate Removal:**
```python
df = df.drop_duplicates().reset_index(drop=True)
```

**c) Outlier Capping (reduces extreme values):**
```python
df[col] = df[col].clip(lower=lower, upper=upper)
```

**d) Correlation-Based Insight (via heatmap):**
The correlation heatmap in `data_loader.py` helps identify highly correlated features that could be candidates for removal to reduce redundancy.

---

## 12. Feature Scaling — Normalization & Standardization

### 12.1 What is Feature Scaling?

Feature scaling transforms numerical features to a **common range or distribution** so that no single feature dominates the model due to its larger magnitude.

### 12.2 Normalization vs Standardization

| Aspect | Normalization (Min-Max) | Standardization (Z-Score) |
|--------|------------------------|--------------------------|
| **Formula** | `(x - min) / (max - min)` | `(x - μ) / σ` |
| **Range** | [0, 1] | No fixed range (mean=0, std=1) |
| **Sensitive to outliers** | Yes | Less so |
| **Best for** | Neural networks, KNN, distance-based | Linear models, tree-based models |
| **sklearn class** | `MinMaxScaler` | `StandardScaler` |

### 12.3 How Our Project Uses Both

**a) Standardization — StandardScaler (`preprocessing.py`):**
```python
def scale_features(df, numerical_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info("Fitted and applied StandardScaler to %d columns",
                    len(numerical_cols))
    else:
        # Use pre-fitted scaler for inference
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df, scaler
```

Applied to **all numerical features** before model training. The scaler is saved and reused during prediction.

**b) Normalization — MinMaxScaler (`feature_engineering.py`):**
```python
def normalize_engineered_features(df):
    engineered_cols = list(COMPOSITE_FEATURES.keys()) + ["OverallSkillScore"]
    existing = [c for c in engineered_cols if c in df.columns]
    if existing:
        scaler = MinMaxScaler()
        df[existing] = scaler.fit_transform(df[existing])
    return df
```

Applied specifically to **engineered composite features** to ensure they are all in the [0, 1] range.

**c) Custom Min-Max Normalization (`feature_engineering.py`):**
```python
def _safe_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)
```
A custom implementation that handles **constant columns** (where min = max) by returning 0.5 instead of dividing by zero.

---

## 13. Feature Engineering

### 13.1 What is Feature Engineering?

Feature engineering is the process of **creating new features** from existing raw data to improve model performance. It is often considered the **most important step** in ML because:
- Raw features may not capture complex patterns
- Domain knowledge can be encoded into meaningful features
- Composite features reduce noise and improve signal

### 13.2 Types of Feature Engineering

| Type | Description | Example |
|------|-------------|---------|
| **Composite features** | Weighted combination of existing features | `AcademicScore = 0.6×CGPA - 0.4×Backlogs` |
| **Interaction features** | Product/ratio of two features | `CGPA × Internships` |
| **Aggregation features** | Sum/mean/count across related features | `Total_Skills = Coding + Communication` |
| **Binning** | Converting continuous to categorical | `Age: Young/Middle/Senior` |
| **Polynomial features** | Higher-order terms | `CGPA²` |

### 13.3 How Our Project Engineers Features

Our project creates **7 composite features** in `feature_engineering.py`:

**Feature Definitions:**
```python
COMPOSITE_FEATURES = {
    "AcademicScore": {
        "columns": ["CGPA", "Backlogs"],
        "weights": [0.6, -0.4],
    },
    "TechnicalScore": {
        "columns": ["Coding_Skills", "Projects", "Aptitude_Test_Score"],
        "weights": [0.50, 0.30, 0.20],
    },
    "ExperienceScore": {
        "columns": ["Internships", "Certifications", "Age"],
        "weights": [0.50, 0.30, 0.20],
    },
    "CommunicationScore": {
        "columns": ["Communication_Skills", "Soft_Skills_Rating"],
        "weights": [0.60, 0.40],
    },
    "ProjectScore": {
        "columns": ["Projects"],
        "weights": [1.0],
    },
    "CodingIndex": {
        "columns": ["Coding_Skills", "Aptitude_Test_Score"],
        "weights": [0.60, 0.40],
    },
}
```

**OverallSkillScore — Master Composite:**
```python
OVERALL_WEIGHTS = {
    "AcademicScore": 0.25,
    "TechnicalScore": 0.25,
    "ExperienceScore": 0.15,
    "CommunicationScore": 0.15,
    "ProjectScore": 0.10,
    "CodingIndex": 0.10,
}
```

**How composite features are calculated:**
```python
def create_composite_features(df):
    for feat_name, spec in COMPOSITE_FEATURES.items():
        cols = spec["columns"]
        weights = spec["weights"]

        # Only use columns that exist in the data
        available = [(c, w) for c, w in zip(cols, weights) if c in df.columns]

        # Normalize each column to [0, 1] before combining
        normalised = pd.DataFrame({
            c: _safe_normalize(df[c].astype(float)) for c in avail_cols
        })

        # Weighted sum
        score = sum(w * normalised[c] for c, w in zip(avail_cols, avail_weights))
        df[feat_name] = _safe_normalize(score).round(4)
    return df
```

### 13.4 Summary of Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `AcademicScore` | 0.6×CGPA − 0.4×Backlogs | Academic performance index |
| `TechnicalScore` | 0.5×Coding + 0.3×Projects + 0.2×Aptitude | Technical ability index |
| `ExperienceScore` | 0.5×Internships + 0.3×Certifications + 0.2×Age | Work experience index |
| `CommunicationScore` | 0.6×Communication + 0.4×SoftSkills | Soft skills index |
| `ProjectScore` | 1.0×Projects | Project portfolio index |
| `CodingIndex` | 0.6×Coding + 0.4×Aptitude | Coding proficiency index |
| `OverallSkillScore` | Weighted combination of all above | Master placement readiness score |

---

## 14. Feature Selection

### 14.1 What is Feature Selection?

Feature selection is the process of **choosing the most relevant features** for the model while discarding irrelevant or redundant ones.

### 14.2 Feature Selection Methods

| Method | Type | Description |
|--------|------|-------------|
| **Filter methods** | Statistical | Correlation, chi-square, mutual information |
| **Wrapper methods** | Model-based | Forward/backward selection, RFE |
| **Embedded methods** | Built into model | Lasso (L1), tree feature importance |
| **Domain knowledge** | Manual | Expert-driven removal |

### 14.3 How Our Project Selects Features

**a) Domain-Based Removal:**
```python
# Student_ID removed — it's an identifier, not a feature
feature_cols = [c for c in df_scaled.columns
                if c != target_col and c != "Student_ID"]
```

**b) Correlation Analysis (`data_loader.py`):**
The correlation heatmap reveals relationships between features. Highly correlated features (|r| > 0.9) could be removed to reduce multicollinearity.

**c) Tree-Based Feature Importance (`explain_prediction.py`):**
```python
# RandomForest provides built-in feature importance
importances = model.feature_importances_
```
The SHAP and feature importance modules rank features by their contribution to predictions, enabling informed feature selection.

**d) Auto-Detection of Column Types (`data_loader.py`):**
```python
def detect_columns(df):
    target_keywords = ["placed", "status", "placement", "target", "label", "hired"]
    # Auto-detect target column by keyword matching
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break
    # Separate numerical and categorical
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    return {"categorical": categorical, "numerical": numerical, "target": target_col}
```

---

## 15. Feature Extraction

### 15.1 What is Feature Extraction?

Feature extraction **transforms raw data into a reduced set of informative features** using mathematical transformations. Unlike feature selection (which picks existing features), feature extraction **creates entirely new representations**.

### 15.2 Common Techniques

| Technique | Domain | Description |
|-----------|--------|-------------|
| **PCA** | General | Reduces dimensions via orthogonal projection |
| **t-SNE** | Visualization | Non-linear embedding for 2D/3D visualization |
| **TF-IDF** | Text | Extracts word importance from documents |
| **Word2Vec** | Text | Dense word embeddings |
| **CNN features** | Images | Learned visual representations |

### 15.3 How Our Project Extracts Features

Our project's feature extraction is achieved through the **composite score construction** in `feature_engineering.py`:

```python
# Raw features are "extracted" into meaningful composite scores
# Example: Multiple raw features → single AcademicScore

# Input: CGPA = 8.5, Backlogs = 0
# Normalized CGPA → 0.85, Normalized Backlogs → 0.0
# AcademicScore = 0.6 × 0.85 + (-0.4) × 0.0 = 0.51
# After final normalization → maps to [0, 1]
```

This is a form of **domain-driven feature extraction** where expert knowledge about what makes a student "placement-ready" is encoded into weighted composite scores.

Additionally, the **SHAP values** computed in `explain_prediction.py` serve as extracted feature representations that explain model decisions:
```python
# SHAP values extract the contribution of each feature to a prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

---

## 16. Preparation of Training & Testing Datasets

### 16.1 What is Train-Test Split?

Splitting data into **training** and **testing** sets is essential to:
- **Train** the model on one subset
- **Evaluate** performance on unseen data
- **Prevent overfitting** — the model shouldn't memorize training data

### 16.2 Common Split Strategies

| Strategy | Description | Ratio |
|----------|-------------|-------|
| **Hold-out** | Simple random split | 80/20 or 70/30 |
| **K-Fold Cross-Validation** | Rotate through K splits | 5 or 10 folds |
| **Stratified Split** | Preserves class balance | Same ratio |
| **Pre-split** | Separate files for train/test | Varies |

### 16.3 How Our Project Prepares Train/Test Data

**a) Pre-Split Files:**
Our dataset comes **pre-split** into two separate CSV files:
```
dataset/train.csv  → 45,000 records (training)
dataset/test.csv   → 5,000 records (testing)
```

**b) Loading Separately (`preprocessing.py`):**
```python
def preprocess_pipeline(df_train, df_test, col_info, output_dir, model_dir):
    # Step 1: Remove duplicates from training data
    df_train = remove_duplicates(df_train)

    # Store sizes
    n_train = len(df_train)

    # Step 2: Concatenate for consistent preprocessing
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Steps 3-6: Missing values → Encoding → Outliers → Scaling
    # ... (applied to combined data)

    # Step 7: Split back into train/test
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_test = y.iloc[n_train:]

    logger.info("Train set: %d, Test set: %d", len(X_train), len(X_test))
```

**c) Cross-Validation for Robust Evaluation (`train_classifier.py`):**
```python
# 5-fold cross-validation on training data
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=cv_folds, scoring="roc_auc", n_jobs=-1
)
```
In addition to the hold-out test set, we use **5-fold cross-validation** to get a more reliable estimate of model performance.

### 16.4 Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PREPROCESSING PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA COLLECTION                                             │
│     └─ Load train.csv and test.csv                              │
│                                                                 │
│  2. DATA CLEANING                                               │
│     ├─ Remove duplicates (training set)                         │
│     └─ Remove Student_ID from features                          │
│                                                                 │
│  3. DATA INTEGRATION                                            │
│     └─ Concatenate train + test for consistent processing       │
│                                                                 │
│  4. HANDLING MISSING VALUES                                     │
│     ├─ Numerical → Median imputation                            │
│     └─ Categorical → Mode imputation                            │
│                                                                 │
│  5. DATA TRANSFORMATION                                         │
│     ├─ Target encoding (Placed=1, Not Placed=0)                 │
│     └─ LabelEncoder for categorical features                    │
│                                                                 │
│  6. DATA REDUCTION                                              │
│     └─ Outlier detection & capping (IQR method)                 │
│                                                                 │
│  7. FEATURE ENGINEERING                                         │
│     ├─ AcademicScore, TechnicalScore, ExperienceScore           │
│     ├─ CommunicationScore, ProjectScore, CodingIndex            │
│     └─ OverallSkillScore (master composite)                     │
│                                                                 │
│  8. FEATURE SCALING                                             │
│     ├─ StandardScaler for all numerical features                │
│     └─ MinMaxScaler for engineered features                     │
│                                                                 │
│  9. SAVE TRANSFORMERS                                           │
│     ├─ scaler.joblib                                            │
│     └─ encoders.joblib                                          │
│                                                                 │
│ 10. TRAIN-TEST SPLIT                                            │
│     ├─ Split back to original train/test sizes                  │
│     └─ Returns X_train, X_test, y_train, y_test                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 16.5 Final Output Dictionary

The `preprocess_pipeline()` function returns:
```python
return {
    "X_train": X_train,           # Training features (DataFrame)
    "X_test": X_test,             # Testing features (DataFrame)
    "y_train": y_train,           # Training labels (Series)
    "y_test": y_test,             # Testing labels (Series)
    "scaler": scaler,             # Fitted StandardScaler
    "encoders": encoders,         # Dict of fitted LabelEncoders
    "feature_names": feature_cols, # List of feature column names
    "numerical_cols": all_numerical,
    "categorical_cols": categorical_cols,
    "df_clean": df,               # Full cleaned DataFrame
}
```

---

## ✅ Summary — Coverage Check

| Topic | Covered? | Where in Project |
|-------|----------|-----------------|
| Google Colab | ✅ | `colab_preprocessing.ipynb`, `accuracy_comparison.ipynb` |
| NumPy | ✅ | All modules — `data_loader.py`, `preprocessing.py`, `train_cluster.py`, etc. |
| Pandas | ✅ | All modules — core data structure throughout |
| Matplotlib | ✅ | `data_loader.py`, `train_cluster.py`, `evaluate_classifier.py`, `explain_prediction.py` |
| Scikit-Learn | ✅ | `preprocessing.py`, `feature_engineering.py`, `train_classifier.py`, `train_cluster.py`, `evaluate_classifier.py` |
| Data Collection | ✅ | `data_loader.py` — CSV loading |
| Data Cleaning | ✅ | `preprocessing.py` — duplicates, outliers |
| Handling Missing Values | ✅ | `preprocessing.py` — median/mode imputation |
| Data Integration | ✅ | `preprocessing.py` — train+test concatenation |
| Data Transformation | ✅ | `preprocessing.py` — LabelEncoder, target mapping |
| Data Reduction | ✅ | `preprocessing.py` — outlier capping, ID removal |
| Feature Scaling (Normalization) | ✅ | `feature_engineering.py` — MinMaxScaler |
| Feature Scaling (Standardization) | ✅ | `preprocessing.py` — StandardScaler |
| Feature Engineering | ✅ | `feature_engineering.py` — 7 composite scores |
| Feature Selection | ✅ | `data_loader.py` — correlation + auto-detection; `explain_prediction.py` — SHAP |
| Feature Extraction | ✅ | `feature_engineering.py` — composite scores; `explain_prediction.py` — SHAP values |
| Train-Test Split | ✅ | `preprocessing.py` — pre-split files + cross-validation |

---

> **📌 Every single topic is implemented in the PlacementIQ AI project with working, production-quality code.**
