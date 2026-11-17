# 09. DATASET TO MODEL PIPELINE

This document explains the complete journey of the PlacementIQ AI project, from the raw CSV datasets all the way to the trained machine learning model. It is based entirely on the **actual Python implementation** inside the `src/` directory.

---

## 1. DATASET SOURCES

The project currently uses a single primary dataset split into training and testing files.

| Dataset | File | Source | Rows | Columns | Purpose |
|---------|------|--------|------|---------|---------|
| Placement Training Data | `dataset/cleaned_train.csv` | Source not documented | 45,000 | 15 | Used to train the ML models and learn placement patterns |
| Placement Testing Data | `dataset/test.csv` | Source not documented | 5,000 | 15 | Used to evaluate the model's accuracy on unseen data |

*Note: If you downloaded this from Kaggle, you can update the source to the exact Kaggle URL.*

---

## 2. ORIGINAL DATASET FEATURES

The original dataset contains the following 15 columns. 

| Feature | Dataset | Data Type | Meaning | Example | Used? |
|---------|---------|-----------|---------|---------|-------|
| `Student_ID` | cleaned_train.csv | Numerical | Unique identifier for each student | 1048 | No |
| `Age` | cleaned_train.csv | Numerical | Age of the student | 22 | Yes |
| `Gender` | cleaned_train.csv | Categorical | Male/Female | Female | Yes |
| `Degree` | cleaned_train.csv | Categorical | Type of degree | B.Tech | Yes |
| `Branch` | cleaned_train.csv | Categorical | Academic specialization | CSE | Yes |
| `CGPA` | cleaned_train.csv | Numerical | Cumulative Grade Point Average | 8.2 | Yes |
| `Internships` | cleaned_train.csv | Numerical | Count of completed internships | 2 | Yes |
| `Projects` | cleaned_train.csv | Numerical | Count of completed projects | 4 | Yes |
| `Coding_Skills` | cleaned_train.csv | Numerical | Self/Test rated coding ability | 8 | Yes |
| `Communication_Skills` | cleaned_train.csv | Numerical | Rated communication ability | 7 | Yes |
| `Aptitude_Test_Score` | cleaned_train.csv | Numerical | Score on logical/aptitude test | 78 | Yes |
| `Soft_Skills_Rating` | cleaned_train.csv | Numerical | Rated interpersonal skills | 7 | Yes |
| `Certifications` | cleaned_train.csv | Numerical | Count of online certificates | 3 | Yes |
| `Backlogs` | cleaned_train.csv | Numerical | Number of uncleared subjects | 0 | Yes |
| `Placement_Status` | cleaned_train.csv | Categorical | Whether they were hired | Placed | **Target** |

---

## 3. DATASET INSPECTION

Immediately after loading the CSV, the system performs inspection in `src/data_loader.py`:

```text
CSV
 ↓
pd.read_csv() (loads dataframe)
 ↓
detect_columns() (auto-detects categorical, numerical, and target columns)
 ↓
missing_value_report() (counts NaN values and percentages)
 ↓
feature_summary() (generates describe() statistics)
 ↓
EDA Visualizations (histograms, boxplots, class distribution, heatmaps)
```

The target column is auto-detected by searching for keywords like "placed", "status", "hired".

---

## 4. WHAT WAS DELETED?

### Features Removed From Original Dataset

| Removed Feature | Original Dataset | Why Removed | Code Location |
|-----------------|------------------|-------------|---------------|
| `Student_ID` | cleaned_train.csv / test.csv | It is a unique identifier. It contains no predictive value and would cause the model to memorize students rather than learn patterns. | `src/data_loader.py` (`detect_columns()`) |

---

## 5. MULTIPLE DATASET MERGING

**No horizontal dataset merge is currently performed.** The project does not combine unrelated datasets based on a foreign key (like joining a student table with a company table).

Instead, the project performs a **Vertical Concatenation** in `src/preprocessing.py`:

| Dataset | Merge Key | Method | Purpose |
|---------|-----------|--------|---------|
| `cleaned_train.csv` + `test.csv` | None | `pd.concat(axis=0)` | Combined temporarily to ensure LabelEncoders and Missing Value Imputers see all possible categories and values, then split back apart. |

---

## 6. WHAT WAS ADDED?

### New / Engineered Features

In `src/feature_engineering.py`, the system creates several composite scores to help the model understand the "bigger picture" of a student.

| New Feature | Input Features | Formula/Logic (Weights) | Why Added | Used in Model? |
|-------------|----------------|-------------------------|-----------|----------------|
| `AcademicScore` | CGPA, Backlogs | 0.60(CGPA) - 0.40(Backlogs) | Combines grades and penalties | Yes |
| `TechnicalScore` | Coding, Projects, Aptitude | 0.50(Coding) + 0.30(Projects) + 0.20(Apt) | Measures hard skills | Yes |
| `ExperienceScore` | Internships, Certs, Age | 0.50(Intern) + 0.30(Certs) + 0.20(Age) | Measures professional exposure | Yes |
| `CommunicationScore`| Comm_Skills, Soft_Skills | 0.60(Comm) + 0.40(Soft) | Measures interpersonal ability | Yes |
| `ProjectScore` | Projects | 1.0(Projects) | Isolates project capability | Yes |
| `CodingIndex` | Coding, Aptitude | 0.60(Coding) + 0.40(Apt) | Measures problem-solving | Yes |
| `OverallSkillScore` | All above composites | Weighted sum of the above | Single metric for readiness | Yes |

*Note: Before applying weights, all input columns are MinMax Normalized to a [0, 1] scale using `_safe_normalize()`.*

---

## 7. FINAL DATASET

Immediately before model training, the dataset has been transformed.

## FINAL DATASET
Rows: 45,000 (train) + 5,000 (test)
Columns: 20 (13 original + 7 engineered)
Target: `Placement_Status`

| # | Final Feature | Original/Engineered | Data Type | Used for Training? |
|---|---------------|---------------------|-----------|--------------------|
| 1-13 | Age, Gender, Degree, etc. | Original | Numerical (Encoded) | Yes |
| 14-20 | AcademicScore, TechnicalScore, etc. | Engineered | Numerical (Scaled) | Yes |

**Transformations Applied:**
- `Student_ID` was removed.
- Categorical columns (`Gender`, `Degree`, `Branch`) were encoded using `LabelEncoder`.
- Target (`Placement_Status`) was mapped to `1` (Placed) and `0` (Not Placed).
- Outliers were capped using the IQR method.
- All numerical columns were scaled using `StandardScaler`.

---

## 8. RAW DATA → FINAL DATASET

```text
RAW KAGGLE DATA (dataset/cleaned_train.csv)
       ↓
Load CSV (data_loader.py)
       ↓
Detect Columns (Categorical vs Numerical vs Target)
       ↓
Remove Duplicates (Only in train set)
       ↓
Handle Missing Values (Median for numbers, Mode for text)
       ↓
Encode Categorical Features (LabelEncoder)
       ↓
Cap Outliers (IQR Method)
       ↓
Feature Engineering (Create 7 new composite scores)
       ↓
Scale Features (StandardScaler)
       ↓
Final Dataset
       ↓
Train/Test Split (Split back to original sizes)
       ↓
Model Training (train_classifier.py)
```

---

## 9. PREPROCESSING BEFORE TRAINING

All preprocessing happens in `src/preprocessing.py`:
1. **Duplicates:** Removed using `df.drop_duplicates()`.
2. **Imputation:** `fillna(median)` for numerical, `fillna(mode)` for categorical.
3. **Outliers:** Values outside `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` are clipped.
4. **Encoding:** `LabelEncoder()` converts strings like "Male" to integers.
5. **Scaling:** `StandardScaler()` shifts data to have mean=0 and std=1.

> [!WARNING]
> **DATA LEAKAGE NOTE:**
> In proper ML pipelines, you must prevent data leakage:
> **Correct:** `scaler.fit(X_train)`, then `scaler.transform(X_train)`, `scaler.transform(X_test)`.
> **Incorrect:** `scaler.fit_transform(Whole_Dataset)`.
> 
> *Currently in the code*, `df_train` and `df_test` are concatenated vertically before scaling. This means the scaler calculates the Mean and Standard Deviation using the Test data too. This is a form of **Data Leakage**, as the model is indirectly "seeing" information about the test set during preprocessing. 

---

## 10. X AND Y

In `src/preprocessing.py`, the features are separated from the target:

```python
feature_cols = [c for c in df_scaled.columns if c != target_col and c != "Student_ID"]
X = df_scaled[feature_cols]
y = df_scaled[target_col]
```

- **X (Input Features):** All numerical and encoded columns (Age, CGPA, AcademicScore, etc.)
- **y (Target Variable):** `Placement_Status` (0 or 1).

They are separated because `X` is the data the model studies, and `y` is the answer sheet it tries to predict.

---

## 11. TRAINING THE MODEL

In `src/train_classifier.py`:

```text
X_train / y_train
 ↓
Train 5 different models (RandomForest, XGBoost, etc.)
 ↓
Cross-Validation (Split train data into 5 folds to test stability)
 ↓
Evaluate on X_test / y_test
 ↓
Compare ROC-AUC Scores
 ↓
Select the Best Model
 ↓
Save to models/best_classifier.joblib
```

---

## 12. MODELS USED

The project trains an ensemble of models and picks the best one automatically.

1. **Random Forest:** Builds multiple decision trees and takes a majority vote. Excellent at preventing overfitting.
2. **Gradient Boosting:** Builds trees sequentially, where each new tree corrects the errors of the previous one.
3. **XGBoost:** An optimized, faster version of Gradient Boosting. Very powerful for tabular data.
4. **Logistic Regression:** A simple linear model. Used as a baseline.
5. **Decision Tree:** A single tree model. Fast but prone to overfitting.

*The model with the highest ROC-AUC score is chosen as the final model.*

---

## 13. HOW PLACEMENT PROBABILITY IS CALCULATED

When a new student is evaluated in `src/predict.py`:

1. The data is passed to the saved model.
2. The code calls `model.predict_proba(student_data)`.
3. This function outputs an array like `[0.15, 0.85]`. 
4. Index `[1]` (0.85) represents the probability of the positive class (Placed).
5. It is multiplied by 100 to get **85.0%**.

The **Readiness Level** is assigned using thresholds (e.g., >80% = Highly Competitive, <40% = Beginner). 
*Note: This probability is a statistical estimate based on historical data, NOT a guarantee of placement.*

---

## 14. MODEL EVALUATION

Models are evaluated using the **ROC-AUC Score** (Receiver Operating Characteristic - Area Under Curve).
- **WHAT it measures:** How well the model separates the "Placed" students from the "Not Placed" students.
- **WHY we use it:** It is better than pure "Accuracy", especially if datasets are imbalanced. 
- **WHAT we achieved:** The cross-validation and test scores are logged in the terminal when training finishes.

---

## 15. EXPLAINABLE AI

Explainable AI tells us *why* the model made a specific prediction.
- **Global Explanation:** In `train_classifier.py`, the system logs Feature Importances (which features matter most across *all* students, e.g., Communication_Skills usually ranks highest).
- **Individual Explanation:** In `src/explain_prediction.py`, the system uses **SHAP** (SHapley Additive exPlanations) to look at a single student and say: "Your placement chance increased by +5% because of your high CGPA, but decreased by -10% because of 0 Internships."

---

## 16. CLUSTERING

In `src/train_cluster.py`, unsupervised learning is used to group students.
- **Algorithm:** K-Means
- **Optimal K:** Selected automatically by testing K=2 to 10 and finding the highest **Silhouette Score**. The Elbow method (inertia) is also plotted.
- **Cluster Interpretation:** Once clusters are formed, the system passes the center of each cluster (centroid) to the classifier to get a placement probability.
- **Naming:** The cluster with the highest probability is named "Top Performers", the lowest is named "Needs Improvement". *These names are just human interpretations of the statistical groupings.*

---

## 17. RECOMMENDATION ENGINE

Recommendations in this project are generated using **Rule-based logic**, NOT Machine Learning.
In `src/recommendation_engine.py`, the code checks if specific scores (like `TechnicalScore`) fall below a manual threshold (e.g., `0.55`).
If it falls below, it triggers a hardcoded array of suggestions (e.g., "Practice 30 DSA problems"). It then assembles these into a 4-week roadmap.

---

## 18. FINAL PREDICTION FLOW

```text
Student Input (Terminal)
       ↓
MinMax Normalization & Composite Score Creation
       ↓
StandardScaler & LabelEncoder (Using saved joblib files)
       ↓
Trained Random Forest Model
       ↓
Placement Probability (e.g. 87.4%) + Readiness Level
       ↓
SHAP (Explains why)
       ↓
K-Means Model (Assigns to "Top Performers" cluster)
       ↓
Recommendation Engine (Identifies weaknesses below threshold)
       ↓
Final Report Printed to Terminal & Saved to TXT
```

---

## 19. FILE-BY-FILE CONNECTION

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `data_loader.py` | Load and inspect data | `cleaned_train.csv` | EDA plots, Column definitions |
| `feature_engineering.py`| Create composite features | Raw DataFrame | DataFrame with 7 new features |
| `preprocessing.py` | Clean, Encode, Scale | Engineered DataFrame | Cleaned `X_train`, `y_train` |
| `train_classifier.py` | Train & pick best model | `X_train`, `y_train` | `best_classifier.joblib` |
| `train_cluster.py` | Train KMeans | `X_train` | `cluster_model.joblib` |
| `predict.py` | Orchestrator / User CLI | User Input via Terminal | Placement Report |

---

## 20. WHAT WE HAVE VS WHAT WE COULD ADD

### CURRENTLY IMPLEMENTED
- Classical ML classification (Random Forest, XGBoost)
- Composite feature engineering
- KMeans Clustering with dynamic naming
- Rule-based 4-week roadmap generation
- SHAP feature importance
- DistilBERT fine-tuning on generated text descriptions

### FUTURE IMPROVEMENTS
- **Actual Resume Parsing:** Currently, the LLM generates a text description from tabular data. Future work could include PyPDF2 to read actual student resume PDFs.
- **GitHub/LeetCode API Integration:** Pull real-time coding scores instead of relying on self-reported inputs.
- **Fix Data Leakage:** Refactor `preprocessing.py` to fit the `StandardScaler` only on the training split before transforming the test split.

---

## PLACEMENTIQ AI IN SIMPLE WORDS

1. We load raw student placement data from a CSV.
2. We remove useless columns like `Student_ID`.
3. We create new composite scores (like combining CGPA and Backlogs into an `AcademicScore`).
4. We fill in missing data and turn text (like "Male") into numbers using Encoders.
5. We scale all numbers so they are on the same playing field.
6. We train 5 different Machine Learning models on the data.
7. We test them and save the one that predicts the most accurately.
8. We also group similar students together using K-Means Clustering.
9. When a new student enters their details, the system prepares their data exactly the same way.
10. The trained model estimates their Placement Probability.
11. SHAP math calculates exactly which skills helped or hurt their score.
12. The rule-based engine checks their weaknesses and generates a 4-week study plan.

---

## QUESTIONS I SHOULD BE ABLE TO ANSWER

**1. Which datasets did you use?**
We used a tabular CSV dataset (`cleaned_train.csv` and `test.csv`) containing 45,000 student records with academic and skill metrics.

**2. Which features did you remove?**
We removed `Student_ID` because it has no predictive value.

**3. Which features did you add?**
We engineered 7 composite features like `TechnicalScore` and `CommunicationScore` by weighting and combining related columns.

**4. What is your target variable?**
`Placement_Status`, representing whether a student was placed (1) or not placed (0).

**5. How did you preprocess the data?**
We imputed missing values, encoded categorical strings, capped outliers using the IQR method, and scaled numbers using `StandardScaler`.

**6. How did you train the model?**
We tested five algorithms (Random Forest, Gradient Boosting, etc.) using Cross-Validation, and automatically saved the one with the highest ROC-AUC score.

**7. How is placement probability calculated?**
By using the `predict_proba()` function of the selected classifier, which outputs a confidence score between 0 and 1.

**8. How does Explainable AI work here?**
We use SHAP values to calculate the exact positive or negative contribution of each feature to the final prediction for an individual student.

**9. How are recommendations generated?**
They are rule-based. If a student's `TechnicalScore` falls below 0.55, a pre-written array of technical study tasks is added to their roadmap.

**10. Can your model guarantee placement?**
No. It provides a statistical estimate of readiness based on historical data patterns.
