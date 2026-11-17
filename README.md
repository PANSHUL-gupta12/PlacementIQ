<div align="center">

# 🎓 PlacementIQ AI
**Continuous Evaluation (CE-2) Platform | Artificial Intelligence & Machine Learning (24CSE0316)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Mlxtend](https://img.shields.io/badge/Mlxtend-FP--Growth-green.svg)](https://rasbt.github.io/mlxtend/)
[![Syllabus Verified](https://img.shields.io/badge/Syllabus-100%25%20Verified-brightgreen.svg)]()

*A comprehensive end-to-end Machine Learning system covering the entire official AIML curriculum across 7 dedicated phases and Jupyter Notebooks.*

</div>

---

## 🌟 Overview & Official Syllabus Mapping

**PlacementIQ AI** is divided into **7 comprehensive educational and engineering phases**, each aligned with the official course syllabus:

1. **Phase 1: Data Preprocessing & EDA** — Ingestion, missing value imputation, IQR outlier detection, label encoding, standard scaling, and 80/20 train/test split.
2. **Phase 2: Supervised Learning (Regression)** — Simple Linear, Multiple Linear, and Polynomial Regression predicting compensation package (`Salary_LPA`) with MAE, MSE, RMSE, and $R^2$ Score.
3. **Phase 3: Supervised Learning (Classification)** — Benchmarking all 6 syllabus classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, Naïve Bayes, and SVM.
4. **Phase 4: Model Evaluation and Validation** — 80/20 split verification, Stratified K-Fold CV, Overfitting/Underfitting diagnostics via complexity curves, Bias-Variance trade-off, and Confusion Matrix breakdown.
5. **Phase 5: Unsupervised Learning (Clustering)** — K-Means (Elbow & Silhouette), Hierarchical Clustering (Ward Dendrogram), and DBSCAN density-based noise filtering for student cohort segmentation.
6. **Phase 6: Association Rule Learning, Recommendation Systems & PCA** — Frequent itemset mining via Apriori & FP-Growth, Market Basket Analysis on student competencies, content-based course recommendations, and PCA Scree plot & 2D projection.
7. **Phase 7: Reinforcement Learning & Emerging Trends** — MDP student preparation environment, tabular Q-Learning convergence, optimal preparation policy, comparative study matrix, and AI ethics/fairness.

---

## 📓 Dedicated Phase Jupyter Notebooks

Each phase has its own self-contained, fully documented, and executable Jupyter Notebook:

| Notebook | Focus Topic & Syllabus Section |
| :--- | :--- |
| [`01_Phase1_Data_Preprocessing_and_EDA.ipynb`](01_Phase1_Data_Preprocessing_and_EDA.ipynb) | Data Cleaning, Missing Values, Outliers, Scaling, 80/20 Split |
| [`02_Phase2_Supervised_Regression.ipynb`](02_Phase2_Supervised_Regression.ipynb) | Linear, Multiple & Polynomial Regression, MAE, MSE, RMSE, $R^2$ |
| [`03_Phase3_Supervised_Classification.ipynb`](03_Phase3_Supervised_Classification.ipynb) | Logistic Regression, KNN, Decision Tree, Random Forest, Naïve Bayes, SVM |
| [`04_Phase4_Model_Evaluation_and_Validation.ipynb`](04_Phase4_Model_Evaluation_and_Validation.ipynb) | Stratified K-Fold, Overfitting Curves, Bias-Variance, Confusion Matrix |
| [`05_Phase5_Unsupervised_Clustering.ipynb`](05_Phase5_Unsupervised_Clustering.ipynb) | K-Means, Hierarchical Dendrogram, DBSCAN, Silhouette Score |
| [`06_Phase6_Association_Rules_and_Recommendations.ipynb`](06_Phase6_Association_Rules_and_Recommendations.ipynb) | Apriori, FP-Growth, Market Basket Analysis, Recommendations, PCA |
| [`07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb`](07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb) | MDP Environment, Q-Learning Simulation, Comparative Study, Ethics |

---

## 🛠️ Quick Start & Execution

### 1. Run Complete Syllabus Automated Test Suite
Verify all 7 phases with a single command:
```bash
python test_full_syllabus_suite.py
```

### 2. Run Full End-to-End Pipeline
Executes preprocessing, feature engineering, classification, salary regression, clustering, association rules, PCA, and Q-learning:
```bash
python src/predict.py --train
```

### 3. Interactive Student Prediction & Package Estimation
Prompts for student metrics and generates a dual-prediction report (Placement Probability + Estimated Package in LPA + 4-Week Roadmap):
```bash
python src/predict.py --interactive
```

---

## 📁 Project Architecture

```text
PlacementIQ_AI/
├── 01_Phase1_Data_Preprocessing_and_EDA.ipynb
├── 02_Phase2_Supervised_Regression.ipynb
├── 03_Phase3_Supervised_Classification.ipynb
├── 04_Phase4_Model_Evaluation_and_Validation.ipynb
├── 05_Phase5_Unsupervised_Clustering.ipynb
├── 06_Phase6_Association_Rules_and_Recommendations.ipynb
├── 07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb
├── test_full_syllabus_suite.py
├── dataset/                     # Contains train.csv (40,000) and test.csv (10,000)
├── docs/                        # Complete viva & evaluation documentation
│   ├── 14_CE_TEST_2_EVALUATION_GUIDE_AND_VIVA.md
│   └── 13_ADVANCED_CURRICULUM_CLUSTERING_ASSOCIATION_RL.md
├── models/                      # Saved .joblib estimators (Classifier, Regressor, Scaler)
├── outputs/                     # Generated charts, scree plots, dendrograms, reports
├── src/                         # Modular Python implementations
│   ├── predict.py               # Main CLI Pipeline Orchestrator
│   ├── train_regressor.py       # Linear, Multiple & Polynomial Regression
│   ├── train_classifier.py      # Logistic, KNN, Tree, Forest, Naive Bayes, SVM
│   ├── model_validation.py      # Bias-Variance, Overfitting curves, Confusion Matrix
│   ├── unsupervised_learning.py # K-Means, Hierarchical Dendrogram, DBSCAN
│   ├── association_and_pca.py   # Apriori, FP-Growth, Recommendations, PCA
│   ├── reinforcement_learning.py# MDP, Q-Learning Agent & Optimal Policy
│   ├── feature_engineering.py   # Composite skill index construction
│   ├── preprocessing.py         # Data cleaning, encoding, scaling
│   └── report_generator.py      # Dual Prediction (Probability + Salary LPA)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 📊 Model Performance Summary (80/20 Test Partition)

| Paradigm | Task / Model | Primary Metric | Score on 10,000 Test Records |
| :--- | :--- | :--- | :---: |
| **Supervised Classification** | Random Forest (Best) | Accuracy / ROC-AUC | **100% / 1.000** |
| **Supervised Classification** | Support Vector Machine (SVM) | Accuracy / ROC-AUC | **99.98% / 1.000** |
| **Supervised Classification** | Decision Tree | Accuracy / ROC-AUC | **100% / 1.000** |
| **Supervised Classification** | Logistic Regression | Accuracy / ROC-AUC | **86.38% / 0.940** |
| **Supervised Regression** | Multiple Linear Regression | $R^2$ / MAE | **0.9912 / 0.362 LPA** |
| **Supervised Regression** | Polynomial Regression (Deg 2) | $R^2$ / MAE | **0.9912 / 0.365 LPA** |
| **Unsupervised Clustering** | K-Means ($K=2$) | Silhouette Score | **0.2552** |
| **Unsupervised Clustering** | Hierarchical (Ward Linkage) | Silhouette Score | **0.1855** |
| **Dimensionality Reduction** | PCA (Top 2 Components) | Explained Variance | **53.32%** |
| **Association Rules** | FP-Growth Frequent Patterns | Rule Count / Lift | **652 Rules / Lift > 1.8** |
| **Reinforcement Learning** | Tabular Q-Learning Agent | Goal Convergence | **Optimal Policy Converged** |
