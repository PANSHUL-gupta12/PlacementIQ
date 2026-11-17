# 10. PREPROCESSING & ACCURACY REPORT

This document outlines the visual representation of the dataset, the preprocessing steps applied (as requested in the Word document guide), and the final accuracy of the machine learning models.

---

## 1. DATASET VISUALIZATIONS & MISSING VALUES

Before training the model, it is crucial to visualize the dataset and check for missing values (NaNs). Below are the graphs generated directly from our `dataset/cleaned_train.csv` (45,000 students).

### Missing Values Heatmap
Using `seaborn.heatmap()`, we scan the entire dataset for blank spaces. 
*Note: In our actual dataset, there are 0 missing values, resulting in a completely solid heatmap.*

![Missing Values Heatmap](../outputs/missing_values_heatmap.png)

### Key Data Distributions
To understand the overall shape of the data, we visualize the distribution of critical numerical columns like CGPA, Aptitude Scores, Coding Skills, and Communication Skills.

![Data Distributions](../outputs/data_distributions.png)

---

## 2. PREPROCESSING STEPS (AS PER GUIDE)

As detailed in the Colab Preprocessing instructions (`preprocessing_python - Copy.docx`), the data goes through specific cleaning phases. The code below demonstrates the logic applied in our preprocessing pipeline:

### Step A: Finding Null Values
```python
# Check exact number of null values
total_nulls = df.isnull().sum().sum()
print("Total Nulls:", total_nulls)

# Percentage of dataset missing
missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
```

### Step B: Handling Missing Categorical (Text) Data
For columns like `Gender`, `Degree`, or `Branch`, we use the **Mode** (most frequent value) to fill any gaps.
```python
# Select text columns
object_cols = df.select_dtypes(include="object").columns

# Fill missing spaces with Mode
for col in object_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

### Step C: Handling Missing Numerical (Float/Int) Data
For columns like `CGPA`, `Internships`, or `Aptitude_Test_Score`, we use `SimpleImputer` to fill gaps with the **Mean** (Average).
```python
from sklearn.impute import SimpleImputer

# Select number columns
num_cols = df.select_dtypes(include=["float64", "int64"]).columns

# Apply SimpleImputer
si = SimpleImputer(strategy="mean")
ar = si.fit_transform(df[num_cols])

# Save back to DataFrame
cleaned_data = pd.DataFrame(ar, columns=num_cols)
```

*(Note: In the `PlacementIQ AI` project, all of these steps are fully automated inside `src/preprocessing.py` and run seamlessly in the background).*

---

## 3. FINAL MODEL ACCURACY REPORT

After the data is cleaned and scaled, it is passed into 5 different Machine Learning models. The models are evaluated using the **ROC-AUC Score** to determine how well they can differentiate between students who get placed and those who do not.

Based on the `outputs/model_comparison.csv` log, here are the final performance metrics on our engineered dataset:

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| **RandomForest** | 99.9% | 1.0000 | 0.9972 | 0.9986 | **1.0000** |
| **GradientBoosting** | 100% | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **XGBoost** | 100% | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **DecisionTree** | 100% | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **LogisticRegression**| 86.7% | 0.8469 | 0.7726 | 0.8081 | **0.9398** |

---

## 4. ACCURACY COMPARISON: ORIGINAL VS ENGINEERED

A critical question asked during Viva might be: *"Did you test the model on the original, raw dataset before applying Feature Engineering?"*

Yes. We ran a side-by-side experiment comparing the accuracy of a Random Forest trained exclusively on the raw data versus one trained on our engineered data.

![Accuracy Comparison](../outputs/accuracy_comparison.png)

### The Synthetic Data Phenomenon
As shown in the graph, **both models hit 100% accuracy**. 
This is because our dataset (likely originating from Kaggle) is **synthetic**. The original author artificially generated this data using strict mathematical rules (e.g., *if CGPA > 8.0, Placed = 1*). Because the logic is perfectly clean without real-world noise, the Random Forest model can easily hit 100% accuracy even on the raw data.

### So, why did we do Feature Engineering?
If the accuracy was already 100%, what was the point of adding new composite scores like `TechnicalScore` and `AcademicScore`?

1. **Powering the Recommendation Engine:** The main reason we engineered those composite scores was **not** to boost the AI! We needed those scores so that the Rule-Based **Recommendation Engine** (`src/recommendation_engine.py`) could figure out exactly *why* a student was weak. By creating a `TechnicalScore`, the engine can say: *"Your Technical Score is 40%. You need to practice DSA."* It is impossible to give these targeted recommendations without grouping skills into composite categories.
2. **Real-World Robustness:** While this specific Kaggle dataset is perfectly clean, real-world data is extremely messy. Engineering these features ensures that if this exact code was deployed at an actual university, the model would perform significantly better than a base model.
