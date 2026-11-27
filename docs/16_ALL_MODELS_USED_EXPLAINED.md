# Complete Guide to All Models Used in PlacementIQ AI

This document provides a complete, up-to-date inventory and explanation of **every Machine Learning, Deep Learning, Unsupervised, Association, Dimensionality Reduction, Reinforcement Learning, and Explainable AI model** used across the **PlacementIQ AI** project.

---

## 1. High-Level Architecture Map

PlacementIQ AI processes **50,000 student profiles** partitioned via an **80/20 stratified split** (`40,000` training / `10,000` test in `dataset/train.csv` and `dataset/test.csv`) through the following model layers:

```text
Raw Student Data (13 Predictive Columns)
   │
   ├─► Feature Engineering & Preprocessing ──► 7 Composite Scores + LabelEncoder + StandardScaler
   │
   ├── 1. Supervised Classification (8 Models) ──► Predicts Placement_Status (Placed / Not Placed)
   ├── 2. Supervised Regression (3 Models)     ──► Predicts Salary_LPA (Expected CTC Package)
   ├── 3. Deep Learning / NLP (DistilBERT)     ──► Classifies Natural-Language Resume Text
   ├── 4. Unsupervised Clustering (3 Models)   ──► Segments Student Cohorts & Detects Outliers
   ├── 5. Dimensionality Reduction (PCA)       ──► Projects 10 Numeric Features onto 2D Space
   ├── 6. Association Mining & Recommenders    ──► Apriori / FP-Growth Skill Rules & Course Matching
   ├── 7. Reinforcement Learning (Q-Learning)  ──► Derives Optimal Study Sequence Policy
   └── 8. Explainable AI (SHAP TreeExplainer)  ──► Explains Per-Student Feature Contributions
```

---

## 2. Supervised Learning — Classification Models (8 Models)

- **Source File:** `src/train_classifier.py`
- **Target Variable:** `Placement_Status` (`1` = Placed, `0` = Not Placed)
- **Input Features:** 20 features (`13` raw profile attributes + `7` engineered composite scores). `Salary_LPA` and `Student_ID` are strictly excluded to prevent target leakage.
- **Saved Winner Artifact:** `models/best_classifier.joblib` (`RandomForestClassifier`) & `models/classifier_meta.joblib`
- **Benchmark CSV:** `outputs/model_comparison.csv`

| # | Model Name | Key Configuration (`src/train_classifier.py`) | How It Works | Verified Test Accuracy | Verified Test ROC-AUC |
|---|---|---|---|---|---|
| 1 | **Random Forest** *(Selected Production Model)* | `n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `random_state=42` | Ensemble of 200 Decision Trees built via bagging & random feature subsets; outputs majority vote / probability average. Resistant to overfitting. | **1.0000** | **1.0000** *(CV: 1.0000)* |
| 2 | **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric="logloss"` | Regularized gradient boosted trees using second-order gradient optimization and parallel split finding. | **1.0000** | **1.0000** *(CV: 1.0000)* |
| 3 | **Gradient Boosting** | `n_estimators=200`, `max_depth=5`, `learning_rate=0.1` | Trains decision trees sequentially so each new tree corrects residual errors of previous trees. | **1.0000** | **1.0000** *(CV: 1.0000)* |
| 4 | **Decision Tree** | `max_depth=8`, `min_samples_split=5` | Learns an interpretable if-else hierarchy of feature splits (e.g., CGPA & backlog cutoffs). | **1.0000** | **1.0000** *(CV: 1.0000)* |
| 5 | **K-Nearest Neighbors (KNN)** | `n_neighbors=5`, `n_jobs=-1` | Non-parametric distance classifier that assigns placement status using the 5 nearest scaled student profiles. | **0.9489** | **0.9843** *(CV: 0.9840)* |
| 6 | **Support Vector Machine (SVM)** | `CalibratedClassifierCV(LinearSVC(max_iter=2000, dual="auto"), cv=3)` | Finds a maximum-margin separating hyperplane and calibrates outputs into placement probabilities. | **0.8585** | **0.9408** *(CV: 0.9401)* |
| 7 | **Logistic Regression** | `max_iter=1000`, `random_state=42` | Linear model wrapped in a sigmoid function outputting probability from weighted feature sums. | **0.8638** | **0.9404** *(CV: 0.9396)* |
| 8 | **Gaussian Naive Bayes** | Default `GaussianNB()` | Probabilistic Bayes-theorem classifier assuming feature conditional independence with Gaussian distributions. | **0.8516** | **0.9229** *(CV: 0.9179)* |

> **Why Tree Ensembles Outperform Linear Models Here:** Placement eligibility follows multi-threshold rules (for example, low backlogs combined with minimum CGPA and coding benchmarks). Tree ensembles (`RandomForest`, `XGBoost`) capture axis-aligned non-linear cutoffs naturally, whereas linear boundaries (`LogisticRegression`, `LinearSVC`) cannot cleanly isolate sharp threshold interactions.

---

## 3. Supervised Learning — Regression Models (3 Models)

- **Source File:** `src/train_regressor.py`
- **Target Variable:** `Salary_LPA` (Continuous annual CTC in Lakhs Per Annum)
- **Saved Winner Artifact:** `models/best_regressor.joblib` (`Multiple Linear Regression`) & `models/regressor_meta.joblib`
- **Benchmark CSV:** `outputs/regression_metrics_comparison.csv`
- **Visualizations:** `outputs/regression_actual_vs_predicted.png`, `outputs/regression_residuals.png`

| # | Model Name | Input Features Used | How It Works | MAE (LPA) | RMSE (LPA) | R² Score |
|---|---|---|---|---|---|---|
| 1 | **Multiple Linear Regression** *(Selected Production Regressor)* | All numeric/engineered predictors | Models `Salary_LPA` as a weighted linear combination across all academic, coding, internship, project, and backlog features simultaneously. | **0.3621** *(5k/1k)* / **0.3646** *(full 40k/10k)* | **0.4535** / **0.4581** | **0.9912** / **0.9910** |
| 2 | **Polynomial Regression (Degree 2)** | Degree-2 interactions inside `Pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=1.0))` | Adds quadratic and pairwise cross-terms with Ridge regularization to test non-linear curvature in package estimation. | **0.3647** / **0.3642** | **0.4559** / **0.4573** | **0.9912** / **0.9910** |
| 3 | **Simple Linear Regression** | Single top predictor (`Coding_Skills`) | Fits `y = mx + b` using only `Coding_Skills`, serving as a single-variable baseline showing why multi-feature regression is necessary. | **2.0425** / **2.0093** | **2.5291** / **2.4800** | **0.7278** / **0.7360** |

---

## 4. Deep Learning / NLP Transformer Model — DistilBERT

- **Source File:** `src/llm_resume_classifier.py`
- **Saved Artifacts:** `models/llm_resume_model/` (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`)
- **Base Architecture:** HuggingFace `DistilBertForSequenceClassification` (`distilbert-base-uncased`, 6 Transformer layers, 2 output classes)
- **Training Setup:** Converts student feature rows into descriptive natural-language resume paragraphs via `generate_resume_text()`, tokenizes (`max_length=128`), and fine-tunes with `AdamW` (`lr=2e-5`, `batch_size=16`, `epochs=3`, `max_samples=5000`, 80/20 train/val split).
- **Role in Project:** Runs alongside `RandomForest` during interactive prediction (`python src/predict.py --interactive`) and standalone (`python src/llm_resume_classifier.py`) to provide an independent NLP assessment (`99.04%` Placed confidence on strong profiles).

---

## 5. Unsupervised Learning — Clustering Models (3 Algorithms)

- **Source Files:** `src/train_cluster.py` (production K-Means pipeline) & `src/unsupervised_learning.py` (multi-algorithm curriculum module)
- **Saved Artifacts:** `models/cluster_model.joblib`, `models/cluster_names.joblib`
- **Outputs:** `outputs/student_cluster_profiles.csv`, `outputs/cluster_elbow_silhouette.png`, `outputs/unsupervised_cluster_evaluation.png`, `outputs/hierarchical_dendrogram.png`

| # | Algorithm | Configuration | Purpose & Verified Behaviour |
|---|---|---|---|
| 1 | **K-Means Clustering** *(Production Segmenter)* | `n_init=10`, `max_iter=300`, `random_state=42`; Auto-selects optimal `K` via Silhouette Score across `K=2..10` (`train_cluster.py`) or `K=2..6` (`unsupervised_learning.py`) | Partitions students around centroids into named cohorts ranked by placement probability. Selects **K=2** (`Top Performers` vs `Needs Improvement`, Silhouette `0.2325`–`0.2552`). |
| 2 | **Hierarchical Agglomerative Clustering** | `n_clusters=4`, `metric="euclidean"`, `linkage="ward"` | Bottom-up clustering minimizing within-cluster variance (Ward linkage) and plotting a truncated **Dendrogram** (`hierarchical_dendrogram.png`, Silhouette `0.0953`). |
| 3 | **DBSCAN** | `eps=2.0` (standalone) / `1.5` (default), `min_samples=15` | Density-based clustering requiring no pre-set `K`; groups dense student profiles and flags atypical anomalies as **noise (`-1`)** (`1` cluster, `102` noise points = `5.10%` on 2,000-sample run). |

---

## 6. Dimensionality Reduction — Principal Component Analysis (PCA)

- **Source File:** `src/association_and_pca.py` (`run_pca_analysis`)
- **Model:** `sklearn.decomposition.PCA` (full PCA for Scree/cumulative variance curve + `PCA(n_components=2)` for 2D visualization)
- **How It Works:** Projects 10 standardized numeric student attributes (`Age`, `CGPA`, `Internships`, `Projects`, `Coding_Skills`, `Communication_Skills`, `Aptitude_Test_Score`, `Soft_Skills_Rating`, `Certifications`, `Backlogs`) onto orthogonal principal axes maximizing retained variance.
- **Verified Outputs:** Top 2 Principal Components capture **`53.32%`** cumulative variance; generates `outputs/pca_variance_and_projection.png` and exports feature eigenvectors to `outputs/pca_feature_loadings.csv`.

---

## 7. Association Rule Mining & Recommendation Systems

- **Source Files:** `src/association_and_pca.py` (`mlxtend`) & `src/recommendation_engine.py`
- **Algorithms Used:**
  1. **Apriori Algorithm** (`mlxtend.frequent_patterns.apriori`) — Level-wise candidate itemset generation across discrete student skill baskets (`CGPA_High`, `Coding_Strong`, `Has_Internship`, `Projects_Rich`, `Multi_Certified`, `Zero_Backlogs`, `Placement_Success`).
  2. **FP-Growth Algorithm** (`mlxtend.frequent_patterns.fpgrowth`) — Prefix-tree (`FP-Tree`) frequent pattern mining without candidate explosion (`min_support=0.15`, `min_confidence=0.50`), mining **`145–149` frequent itemsets** and **`650–652` association rules** saved to `outputs/placement_association_rules.csv`.
  3. **Content-Based Recommender & Roadmap Engine:**
     - Skill-weakness course matcher (`COURSE_CATALOG` courses `C01`–`C06` in `association_and_pca.py`).
     - Personalized 4-Week Improvement Roadmap & Skill-Gap Analyser (`recommendation_engine.py`), paired with rule-based ATS Resume Scorer (`resume_scoring.py`), Mock Interview Scorer (`interview_scoring.py`), and Company Tier Matcher (`company_matching.py`).

---

## 8. Reinforcement Learning — Q-Learning Policy Agent

- **Source File:** `src/reinforcement_learning.py`
- **Environment (`StudentPlacementEnv`):** Custom 5-state Markov Decision Process (`Beginner` → `Coding Enthusiast` → `Project Builder` → `Well-Rounded Candidate` → terminal `Placement-Ready Pro`) with 4 preparation actions and stochastic transitions (`-1.0` step penalty, `+50.0` goal reward).
- **Agent (`QLearningPlacementAgent`):** Tabular Q-Learning (`alpha=0.1`, `gamma=0.95`, `epsilon=1.0` decaying by `0.995` to `0.05`) trained for `600–1000` episodes using Bellman updates.
- **Converged Optimal Policy (`outputs/rl_optimal_policy.csv`):**
  1. **Beginner** → *Practice LeetCode / DSA Problems*
  2. **Coding Enthusiast** → *Develop Portfolio Project*
  3. **Project Builder** → *Soft Skills BootCamp*
  4. **Well-Rounded Candidate** → *Mock Technical/HR Interviews*

---

## 9. Preprocessing, Feature Engineering & Explainable AI (XAI)

- **Preprocessing (`src/preprocessing.py`):**
  - **IQR Outlier Capping:** Caps extreme values to `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` without dropping student rows.
  - **`LabelEncoder` (`models/encoders.joblib`):** Encodes `Gender`, `Degree`, and `Branch`.
  - **`StandardScaler` (`models/scaler.joblib`):** Standardizes all 20 predictor features (`mean=0, std=1`) fitted strictly on training data.
- **Composite Feature Engineering (`src/feature_engineering.py`):** Builds 7 normalized `[0, 1]` domain indicators: `AcademicScore`, `TechnicalScore`, `ExperienceScore`, `CommunicationScore`, `ProjectScore`, `CodingIndex`, and `OverallSkillScore`.
- **Model Validation (`src/model_validation.py`):** Runs 5-Fold K-Fold (`0.9994`) vs Stratified K-Fold (`0.9980`), Decision Tree depth curves (`max_depth=1..14` for Underfitting/Overfitting), Bias-Variance decomposition, and Type I / Type II Confusion Matrix plots.
- **Explainable AI (`src/explain_prediction.py`):** Combines global `feature_importances_` (`outputs/feature_importance.png`) and **SHAP `TreeExplainer`** (`outputs/shap_summary.png`) to explain positive and negative feature drivers for every predicted student.

---

## 10. Quick Summary of Saved Model Files (`models/`)

| File / Folder | Model / Artifact Stored | Used For |
|---|---|---|
| `models/best_classifier.joblib` | `RandomForestClassifier` (200 trees, depth 10) | Predicting `Placement_Status` & probability |
| `models/best_regressor.joblib` | `LinearRegression` (Multiple Linear Regression) | Estimating continuous `Salary_LPA` package |
| `models/cluster_model.joblib` | `KMeans` (`K=2`) | Assigning student segment (`Top Performers` / `Needs Improvement`) |
| `models/llm_resume_model/` | Fine-tuned `DistilBERT` Transformer | Classifying natural-language resume text |
| `models/scaler.joblib` | `StandardScaler` (20 features) | Scaling inputs identically during inference |
| `models/encoders.joblib` | `dict[LabelEncoder]` | Encoding `Gender`, `Degree`, `Branch` |
| `models/feature_names.joblib` | Ordered 20-feature list | Enforcing exact feature column alignment |
| `models/classifier_meta.joblib` / `regressor_meta.joblib` | Model name, metrics & `scaled_input` flag | Metadata verification & inference routing |
