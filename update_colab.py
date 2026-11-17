# pyrefly: ignore [missing-import]
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text0 = """# PlacementIQ Data Preprocessing Pipeline (Google Colab Version)
This notebook implements the complete data preprocessing pipeline for the PlacementIQ AI project, ready to run on Google Colab."""

code1 = """# Step 1: Mount Google Drive and Load Data
from google.colab import drive
import pandas as pd
import numpy as np
import os

drive.mount('/content/drive')

# Update this path to where you uploaded the dataset in your Google Drive
data_path = '/content/drive/MyDrive/PlacementIQ/dataset/placement.csv'
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
display(df.head())"""

text2 = "## Step 2: Duplicate Removal\nRemove any duplicate rows in the dataset."

code2 = """before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows. New shape: {df.shape}")"""

text3 = "## Step 3: Missing Value Handling\nImpute numerical columns with median and categorical columns with mode."

code3 = """# Detect numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if 'Student_ID' in numerical_cols:
    numerical_cols.remove('Student_ID')
    
target_col = 'Placement_Status'
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# Impute Numerical
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Impute Categorical
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])
        
print("Remaining Missing Values:", df.isnull().sum().sum())"""

text4 = "## Step 4: Target Encoding\nEncode the target column (Placement_Status)."

code4 = """if df[target_col].dtype == object or str(df[target_col].dtype) in ["string", "category", "str"]:
    unique_vals = set(df[target_col].dropna().unique())
    if unique_vals == {"Placed", "Not Placed"}:
        df[target_col] = df[target_col].map({"Placed": 1, "Not Placed": 0})
        print("Mapped 'Placed'->1, 'Not Placed'->0")
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        print("Target Encoded classes:", le.classes_)"""

text5 = "## Step 5: Categorical Feature Encoding\nEncode categorical features using LabelEncoder."

code5 = """from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Encoded {col} - classes: {le.classes_}")

all_numerical = numerical_cols + categorical_cols"""

text6 = "## Step 6: Outlier Detection and Capping (IQR Method)"

code6 = """total_capped = 0
for col in all_numerical:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        total_capped += outliers
        
print(f"Total outliers capped: {total_capped}")"""

text7 = "## Step 7: Feature Scaling\nStandardize numerical features."

code7 = """from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[all_numerical] = scaler.fit_transform(df[all_numerical])
print("StandardScaler applied to features.")"""

text8 = "## Step 8: Train/Test Split and Save"

code8 = """from sklearn.model_selection import train_test_split
import joblib

# Prepare X and y
feature_cols = [c for c in df.columns if c != target_col and c != "Student_ID"]
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

# Save outputs to Google Drive
output_dir = '/content/drive/MyDrive/PlacementIQ/outputs'
model_dir = '/content/drive/MyDrive/PlacementIQ/models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
joblib.dump(encoders, os.path.join(model_dir, "encoders.joblib"))

print("Cleaned data and transformers saved successfully to Google Drive!")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text0),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(text7),
    nbf.v4.new_code_cell(code7),
    nbf.v4.new_markdown_cell(text8),
    nbf.v4.new_code_cell(code8)
]

with open('colab_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
