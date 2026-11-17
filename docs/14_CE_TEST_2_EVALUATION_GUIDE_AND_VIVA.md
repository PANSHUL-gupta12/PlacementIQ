# 🎓 Continuous Evaluation (CE Test 2) Master Guide & Viva Preparation
**Course Code:** 24CSE0316 | **Course Title:** Artificial Intelligence & Machine Learning  
**Programme & Semester:** B.E. (CSE) 5th Sem | **Total Marks:** 20 | **Duration:** 120 Mins  
**Course Coordinator:** Dr. Priyanka Gupta | **Project Title:** PlacementIQ AI  

---

## 🎯 Assessment Weightage Breakdown (20 Marks Total)
| Evaluation Component | Weightage | Strategy & Deliverables in PlacementIQ AI |
| :--- | :---: | :--- |
| **Viva** | **7 Marks** | Deep conceptual grasp of all 7 syllabus phases, mathematical formulations, algorithmic trade-offs, and parameter meanings (Section 3 of this document). |
| **PPT (Presentation)** | **7 Marks** | Structured presentation covering problem statement, architecture, regression vs classification results, clustering, association rules, and RL. |
| **Project Idea** | **3 Marks** | Real-world novelty: An end-to-end recruitment intelligence platform that predicts placement eligibility, forecasts salary package (LPA), segments cohort weaknesses, and formulates RL-guided study roadmaps. |
| **Team Performance** | **3 Marks** | Clean modular codebase in `src/`, automated test suites (`test_full_syllabus_suite.py`), 7 self-contained Jupyter notebooks, and reproducible artifacts. |

---

## 📑 The 7 Project Phases & Dedicated Jupyter Notebooks

### 🔹 Phase 1: Data Preprocessing & Exploratory Data Analysis (EDA)
- **Notebook:** [`01_Phase1_Data_Preprocessing_and_EDA.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/01_Phase1_Data_Preprocessing_and_EDA.ipynb)
- **Key Concepts:**
  - Ingestion of 50,000 student records.
  - Missing value imputation: Numerical median and categorical mode.
  - Label encoding for nominal variables (`Gender`, `Degree`, `Branch`).
  - Outlier detection and capping using the Interquartile Range (IQR) method: $[\text{Q1} - 1.5 \cdot \text{IQR}, \;\text{Q3} + 1.5 \cdot \text{IQR}]$.
  - Feature normalization using `StandardScaler` to zero mean and unit variance.
  - **Train-Test Split:** Strict 80% Training ($40,000$) and 20% Testing ($10,000$).

---

### 🔹 Phase 2: Supervised Learning – Regression
- **Notebook:** [`02_Phase2_Supervised_Regression.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/02_Phase2_Supervised_Regression.ipynb)
- **Module:** [`src/train_regressor.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/train_regressor.py)
- **Target Variable:** `Salary_LPA` (Continuous compensation package in Lakhs Per Annum).
- **Models Implemented & Benchmarked:**
  1. **Simple Linear Regression:** Ordinary Least Squares on primary technical predictor (`Coding_Skills` vs `Salary_LPA`).
  2. **Multiple Linear Regression:** Multi-variable modeling using all academic, technical, and internship metrics.
  3. **Polynomial Regression (Degree 2):** Captures non-linear compounding returns on skill combinations.
- **Evaluation Metrics (Tested on 10,000 Unseen Records):**
  - **MAE (Mean Absolute Error):** $0.3621$ LPA (Average absolute prediction discrepancy).
  - **MSE (Mean Squared Error):** $0.2056$.
  - **RMSE (Root Mean Squared Error):** $0.4535$ LPA.
  - **$R^2$ Score (Coefficient of Determination):** $0.9912$ ($99.12\%$ of salary variance explained).

---

### 🔹 Phase 3: Supervised Learning – Classification
- **Notebook:** [`03_Phase3_Supervised_Classification.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/03_Phase3_Supervised_Classification.ipynb)
- **Module:** [`src/train_classifier.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/train_classifier.py)
- **Target Variable:** `Placement_Status` (Binary: Placed $[1]$ vs. Not Placed $[0]$).
- **All 6 Syllabus Classifiers Benchmarked:**
  1. **Logistic Regression:** Log-odds modeling via Sigmoid activation ($86.38\%$ accuracy, $0.9404$ ROC-AUC).
  2. **K-Nearest Neighbour (KNN):** Distance-weighted neighborhood voting ($K=5$).
  3. **Decision Tree:** Gini impurity recursive binary splitting.
  4. **Random Forest:** Ensemble bagging with feature subspace randomization ($100\%$ accuracy, $1.000$ ROC-AUC).
  5. **Naïve Bayes:** Probabilistic classifier applying Bayes' theorem with feature independence assumption.
  6. **Support Vector Machine (SVM):** Maximum-margin separation hyperplane with calibrated probabilities.

---

### 🔹 Phase 4: Model Evaluation and Validation
- **Notebook:** [`04_Phase4_Model_Evaluation_and_Validation.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/04_Phase4_Model_Evaluation_and_Validation.ipynb)
- **Module:** [`src/model_validation.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/model_validation.py)
- **Key Concepts & Deliverables:**
  - **Train-Test Split Validation:** Demonstrating generalization on $10,000$ test cases ($6,375$ Not Placed, $3,625$ Placed).
  - **Cross-Validation Comparison:** Standard 5-Fold vs. 5-Fold Stratified CV to prevent class imbalance skew.
  - **Overfitting & Underfitting Diagnosis:** Validation curves plotting model complexity (tree depth 1 to 15) against training vs. validation accuracy.
  - **Bias-Variance Trade-off:** Mathematical error decomposition: $\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$.
  - **Confusion Matrix Breakdown:** Detailed tracking of TP ($3,625$), TN ($6,375$), FP (Type I Error = $0$), and FN (Type II Error = $0$).
  - **Metrics Computed:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.

---

### 🔹 Phase 5: Unsupervised Learning – Clustering
- **Notebook:** [`05_Phase5_Unsupervised_Clustering.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/05_Phase5_Unsupervised_Clustering.ipynb)
- **Module:** [`src/unsupervised_learning.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/unsupervised_learning.py)
- **Key Algorithms:**
  1. **K-Means Clustering:** WCSS (Inertia) minimization; Elbow method and Silhouette optimization across $K \in [2, 6]$.
  2. **Hierarchical Clustering (Agglomerative):** Bottom-up clustering with Ward linkage; dendrogram visualization saved to [`outputs/hierarchical_dendrogram.png`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/outputs/hierarchical_dendrogram.png).
  3. **DBSCAN:** Density-based clustering with $\epsilon=2.2, \text{MinPts}=15$; detects dense clusters while flagging isolated outliers/noise points.
  4. **Multi-Metric Evaluation:** Silhouette Score ($0.2552$), Davies-Bouldin Index ($1.4431$), Calinski-Harabasz Index ($881.10$).
  5. **Application:** Student cohort segmentation into actionable readiness profiles ("Top Performers", "Needs Remedial Coding").

---

### 🔹 Phase 6: Association Rule Learning, Recommendation Systems & PCA
- **Notebook:** [`06_Phase6_Association_Rules_and_Recommendations.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/06_Phase6_Association_Rules_and_Recommendations.ipynb)
- **Module:** [`src/association_and_pca.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/association_and_pca.py)
- **Key Concepts:**
  - **Frequent Itemset Mining:** Transforming student academic and technical profiles into transaction baskets.
  - **Apriori vs. FP-Growth:** Mining $149$ frequent itemsets and $652$ association rules. FP-Growth runs orders of magnitude faster by eliminating candidate generation via compact FP-Trees.
  - **Metrics:** Support, Confidence, Lift ($> 1.8$ for high coding + multiple certifications $\implies$ Placement Success).
  - **Recommendation Systems:** Content-based cosine similarity mapping student weakness vectors to customized technical courses.
  - **Dimensionality Reduction (PCA):** Covariance matrix eigen-decomposition; Scree Plot and 2D student space projection explaining over $53\%$ of variance in just 2 principal components.

---

### 🔹 Phase 7: Reinforcement Learning & Emerging Trends
- **Notebook:** [`07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb)
- **Module:** [`src/reinforcement_learning.py`](file:///d:/01_Projects/01_AimlProject/PlacementIQ_AI/src/reinforcement_learning.py)
- **Key Concepts:**
  - **MDP Framework:** Agent (Student), Environment (`StudentPlacementMDP`), States (Beginner, DSA Foundation, Project Portfolio, Interview Ready, Placed), Actions (LeetCode, Projects, Soft Skills, Mock Interviews).
  - **Bellman Equation & Q-Learning:**
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - **Convergence:** Rolling cumulative rewards demonstrating policy convergence to optimal placement preparation order.
  - **Comparative Study:** Comprehensive matrix comparing Supervised, Unsupervised, and Reinforcement Learning across inputs, goals, feedback, and algorithms.
  - **Real-World Applications:** Robotics (locomotion, arm control), Gaming (AlphaGo, Dota 2), Autonomous Vehicles (Tesla FSD trajectory planning).
  - **Ethics & Trends:** Algorithmic bias audits, Explainable AI (XAI), and Generative AI / Foundation Models for resume optimization.

---

## 🎤 Top Viva Questions & Answers (7 Marks Guarantee)

### Q1. What is the fundamental difference between Regression and Classification?
**Answer:**
- **Regression** predicts a **continuous numerical value** ($y \in \mathbb{R}$). In our project, Phase 2 predicts the student's expected salary/CTC package (e.g., $18.65$ LPA) using metrics like MAE, MSE, RMSE, and $R^2$.
- **Classification** predicts a **discrete categorical class** ($y \in \{0, 1\}$). In our project, Phase 3 predicts whether the student is "Placed" or "Not Placed" evaluated via Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Q2. How do you interpret MAE, RMSE, and $R^2$ Score in your regression model?
**Answer:**
- **MAE (Mean Absolute Error):** Represents the average magnitude of errors in LPA without penalizing large errors excessively ($0.3621$ LPA in our Multiple Linear Regression).
- **RMSE (Root Mean Squared Error):** Takes the square root of squared errors ($0.4535$ LPA). It gives disproportionate weight to large outliers, making it sensitive to large mispredictions.
- **$R^2$ Score (Coefficient of Determination):** Measures the proportion of variance in the dependent variable explained by the input features ($0.9912$ in our model, meaning $99.12\%$ of salary variation is captured).

### Q3. Explain the Bias-Variance Trade-off. How did you diagnose it?
**Answer:**
$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$
- **High Bias (Underfitting):** The model is overly simplistic (e.g., shallow decision tree depth $< 3$) and fails to capture underlying patterns, showing high error on both training and test data.
- **High Variance (Overfitting):** The model is excessively complex (e.g., unpruned tree depth $> 10$) and memorizes noise in the training set, achieving high training accuracy but poor validation accuracy.
- **Diagnosis:** In Phase 4, we plotted **Validation Curves** across decision tree depths from 1 to 15, identifying the optimal depth ($4 \le \text{depth} \le 7$) where validation accuracy peaks before dropping due to variance.

### Q4. What is the difference between K-Fold and Stratified K-Fold Cross-Validation?
**Answer:**
- **Standard K-Fold** randomly partitions data into $K$ splits without regarding target class proportions. If a class is rare, some folds may receive very few positive examples.
- **Stratified K-Fold** guarantees that every fold contains approximately the exact same percentage of target samples as the complete dataset (e.g., $63.75\%$ Not Placed and $36.25\%$ Placed in every fold).

### Q5. Why is FP-Growth faster than the Apriori algorithm?
**Answer:**
- **Apriori** uses a candidate generation-and-test approach ($C_k \rightarrow L_k$). It requires multiple full passes over the database to compute support for combinations, which is computationally expensive for large datasets.
- **FP-Growth (Frequent Pattern Growth)** constructs a compact tree structure called an **FP-Tree** in just two passes over the data. It mines frequent itemsets directly by exploring conditional pattern bases without generating any candidate itemsets.

### Q6. How does DBSCAN differ from K-Means Clustering?
**Answer:**
1. **Cluster Shape:** K-Means assumes spherical, convex clusters around mean centroids; DBSCAN discovers clusters of arbitrary geometric shape.
2. **Number of Clusters ($K$):** K-Means requires specifying $K$ upfront; DBSCAN determines the number of clusters automatically based on density parameters ($\epsilon$ and $\text{MinPts}$).
3. **Outlier / Noise Handling:** K-Means forces every outlier into a cluster; DBSCAN explicitly identifies isolated low-density points as noise (labeled $-1$).

### Q7. Explain the Bellman Optimality Equation in Q-Learning.
**Answer:**
$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')$$
In Q-Learning (model-free TD control), the Q-value update rule is:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
Where:
- $\alpha$: Learning rate ($0 < \alpha \le 1$) controlling how much new feedback overrides old estimates.
- $\gamma$: Discount factor ($0 \le \gamma < 1$) determining the importance of future delayed rewards vs. immediate rewards.
- $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$: The **Temporal Difference (TD) Error**.

