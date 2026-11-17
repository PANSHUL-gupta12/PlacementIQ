import json
import os

def create_notebook(filename, cells_content):
    cells = []
    for cell_type, source in cells_content:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

# --- 02_DataCleaning.ipynb ---
cleaning_cells = [
    ("markdown", "# PlacementIQ AI - Data Cleaning\nIn this notebook, we clean the raw dataset, handle missing values, and visualize & cap outliers using Boxplots."),
    ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\n\nos.makedirs('outputs', exist_ok=True)"),
    ("markdown", "## 1. Load Data"),
    ("code", "df = pd.read_csv('dataset/train.csv')\nprint(f'Original shape: {df.shape}')\ndf.head()"),
    ("markdown", "## 2. Drop Irrelevant Columns & Duplicates"),
    ("code", "if 'Student_ID' in df.columns:\n    df.drop(columns=['Student_ID'], inplace=True)\ndf.drop_duplicates(inplace=True)\nprint(f'Shape after dropping duplicates and ID: {df.shape}')"),
    ("markdown", "## 3. Handle Missing Values"),
    ("code", "print('Missing values before imputation:')\nprint(df.isnull().sum())\n\n# Impute numerical with Median, categorical with Mode\nnum_cols = df.select_dtypes(include=['float64', 'int64']).columns\ncat_cols = df.select_dtypes(include=['object']).columns\n\nfor col in num_cols:\n    df[col].fillna(df[col].median(), inplace=True)\nfor col in cat_cols:\n    df[col].fillna(df[col].mode()[0], inplace=True)\n\nprint('\\nMissing values after imputation:')\nprint(df.isnull().sum())"),
    ("markdown", "## 4. Feature Boxplots (Outlier Detection)"),
    ("code", "plt.figure(figsize=(15, 12))\nfeatures_to_plot = [col for col in num_cols if col != 'Placement_Status']\nfor i, col in enumerate(features_to_plot, 1):\n    plt.subplot(3, 4, i)\n    sns.boxplot(y=df[col], color='purple')\n    plt.title(col)\n    plt.ylabel('')\nplt.tight_layout()\nplt.suptitle('Feature Boxplots', y=1.02, fontsize=16, fontweight='bold')\nplt.savefig('outputs/boxplots.png', bbox_inches='tight')\nplt.show()"),
    ("markdown", "## 5. Cap Outliers using IQR"),
    ("code", "for col in features_to_plot:\n    Q1 = df[col].quantile(0.25)\n    Q3 = df[col].quantile(0.75)\n    IQR = Q3 - Q1\n    lower_bound = Q1 - 1.5 * IQR\n    upper_bound = Q3 + 1.5 * IQR\n    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])\n    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])\nprint('Outliers capped successfully.')\ndf.to_csv('dataset/cleaned_train.csv', index=False)")
]

# --- 03_FeatureEngineering.ipynb ---
engineering_cells = [
    ("markdown", "# PlacementIQ AI - Feature Engineering & EDA\nIn this notebook, we visualize the target distribution, engineer composite features, encode categoricals, and plot the correlation heatmap."),
    ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import LabelEncoder, MinMaxScaler"),
    ("markdown", "## 1. Load Cleaned Data"),
    ("code", "df = pd.read_csv('dataset/cleaned_train.csv')\ndf.head()"),
    ("markdown", "## 2. Target Class Distribution"),
    ("code", "plt.figure(figsize=(8, 6))\nax = sns.countplot(data=df, x='Placement_Status', palette=['#2ecc71', '#e74c3c'])\nplt.title('Class Distribution — Placement_Status', fontsize=16, fontweight='bold')\nplt.ylabel('Count', fontsize=14)\nplt.xlabel('Placement_Status', fontsize=14)\nfor p in ax.patches:\n    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),\n                ha='center', va='center', fontsize=14, color='black', xytext=(0, 5), textcoords='offset points', fontweight='bold')\nplt.savefig('outputs/class_distribution.png', bbox_inches='tight')\nplt.show()"),
    ("markdown", "## 3. Feature Engineering"),
    ("code", "df['AcademicScore'] = (df['CGPA'] / 10.0) * 0.6 - (df['Backlogs'] / 5.0) * 0.4\ndf['TechnicalScore'] = (df['Coding_Skills'] + df['Projects'] + (df['Aptitude_Test_Score']/10.0)) / 30.0\nprint('Engineered composite scores successfully.')"),
    ("markdown", "## 4. Categorical Encoding"),
    ("code", "le = LabelEncoder()\ncat_cols = df.select_dtypes(include=['object']).columns\nfor col in cat_cols:\n    if col != 'Placement_Status':\n        df[col] = le.fit_transform(df[col])\n\n# Map target\ndf['Placement_Status'] = df['Placement_Status'].map({'Placed': 1, 'Not Placed': 0})"),
    ("markdown", "## 5. Feature Correlation Heatmap"),
    ("code", "plt.figure(figsize=(12, 10))\ncorr = df.corr()\nmask = np.triu(np.ones_like(corr, dtype=bool))\nsns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)\nplt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')\nplt.savefig('outputs/correlation_heatmap.png', bbox_inches='tight')\nplt.show()")
]

create_notebook('02_DataCleaning.ipynb', cleaning_cells)
create_notebook('03_FeatureEngineering.ipynb', engineering_cells)
print("Notebooks updated successfully!")
