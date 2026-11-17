# Viva Questions & Answers — PlacementIQ AI

This document contains 60+ potential viva questions that can be asked about this project, grouped by category, with detailed answers.

---

## Category 1: Project Overview (10 Questions)

### Q1. What is PlacementIQ AI?
**A:** PlacementIQ AI is an end-to-end machine learning system that predicts a student's campus placement readiness. Given a student's academic, technical, and extracurricular profile, it outputs a placement probability, readiness level, student cluster, skill gap analysis, company recommendations, and a personalised 4-week improvement roadmap.

### Q2. What problem does this project solve?
**A:** Every year, millions of engineering students prepare for campus placements without knowing where they stand. PlacementIQ AI solves this by training ML models on 50,000 historical student records to learn the patterns that distinguish placed students from unplaced ones, and then providing actionable, personalised feedback to new students.

### Q3. What is the technology stack?
**A:** Python 3.10+, scikit-learn for classical ML models, XGBoost for gradient boosting, PyTorch + HuggingFace Transformers for the DistilBERT LLM, SHAP for explainable AI, Pandas/NumPy for data processing, and Matplotlib/Seaborn for visualisations.

### Q4. How many models does the system use?
**A:** It uses 7 models:
- 5 supervised classifiers (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, XGBoost)
- 1 unsupervised clustering model (KMeans)
- 1 deep learning NLP model (DistilBERT for resume classification)

### Q5. What is the dataset used?
**A:** We use a dataset of 50,000 student records split into cleaned_train.csv (45,000 records) and test.csv (5,000 records). Each record has 13 input features (Age, Gender, Degree, Branch, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications, Backlogs) and 1 target variable (Placement_Status: "Placed" or "Not Placed").

### Q6. What makes this project different from a simple binary classifier?
**A:** Instead of just saying "Placed" or "Not Placed", our system provides:
- An exact probability percentage
- A readiness level (Beginner to Exceptional)
- Cluster-based student grouping
- SHAP-based explainable AI showing WHY the model made its prediction
- Skill gap analysis with specific weaknesses identified
- A 4-week personalised roadmap
- Company tier recommendations
- Resume ATS analysis + LLM-based classification
- Interview readiness scoring

### Q7. What are the three modes of running the project?
**A:** (1) `--train` to retrain all ML models from the CSV data, (2) `--interactive` to predict for a single student via terminal prompts, (3) `--batch <file.csv>` to predict for an entire CSV file of students.

### Q8. What output files does the system generate?
**A:** Saved models (`.joblib` files in `models/`), EDA visualisation plots (`.png` in `outputs/`), a model comparison chart, SHAP feature importance plot, a cleaned dataset CSV, and the final placement report (`.txt`).

### Q9. Can this system be deployed in production?
**A:** Yes. The modular architecture (each module is a separate Python file with clean APIs) makes it deployable as a REST API (using Flask/FastAPI), a web application, or embedded in a university's placement portal.

### Q10. What is the accuracy of the best model?
**A:** The Random Forest classifier achieved an ROC-AUC of 1.0 (perfect) on the test set, with 100% precision, recall, and F1-score. This indicates the training data has very clear, linearly separable patterns.

---

## Category 2: Machine Learning Concepts (15 Questions)

### Q11. What is supervised learning?
**A:** Supervised learning is when we train a model using labelled data — i.e., data where we know the correct answer (placed or not placed). The model learns the mapping from input features to the target label.

### Q12. What is unsupervised learning?
**A:** Unsupervised learning finds patterns in data WITHOUT labelled answers. We use KMeans clustering, which groups students into clusters based on similarity without knowing their placement outcome.

### Q13. What is the difference between classification and regression?
**A:** Classification predicts categorical outcomes (e.g., "Placed" or "Not Placed"). Regression predicts continuous values (e.g., predicting a salary amount). Our project uses classification.

### Q14. Explain how Random Forest works.
**A:** Random Forest creates an ensemble of many Decision Trees. Each tree is trained on a random subset of the data and a random subset of features. For classification, the final prediction is the majority vote across all trees. This reduces overfitting and increases accuracy. We use 200 trees with max depth of 15.

### Q15. What is overfitting and how does Random Forest prevent it?
**A:** Overfitting is when a model memorises the training data but fails on new data. Random Forest prevents overfitting through: (1) bagging — each tree sees a different data subset, (2) feature randomness — each split considers a random feature subset, (3) ensemble averaging — individual tree errors cancel out.

### Q16. What is ROC-AUC and why is it our primary metric?
**A:** ROC-AUC is the Area Under the Receiver Operating Characteristic curve. It measures how well the model ranks positive cases above negative cases across all possible thresholds. AUC = 1.0 means perfect ranking, 0.5 means random guessing. We prefer AUC over accuracy because it's threshold-independent and robust to class imbalance.

### Q17. Explain precision and recall.
**A:** Precision = "Of all students I predicted as Placed, how many actually were?" (TP / (TP + FP)). Recall = "Of all students who were actually Placed, how many did I correctly identify?" (TP / (TP + FN)). Both are important — precision reduces false alarms, recall reduces missed placements.

### Q18. What is cross-validation and why do we use it?
**A:** Cross-validation splits the training data into K folds, trains on K-1 folds, tests on the remaining fold, and repeats K times. We use 5-fold stratified CV to: (1) get a reliable estimate of model performance, (2) check if the model is stable across different data splits, (3) detect overfitting.

### Q19. What is feature engineering?
**A:** Feature engineering is the process of creating new, informative features from existing raw data. We create composite features like AcademicScore (combining CGPA and Backlogs) because these capture higher-level concepts that raw features alone don't express.

### Q20. What is feature scaling and why is it necessary?
**A:** Feature scaling transforms all features to a similar range. We use StandardScaler (mean=0, std=1). Without scaling, features with large ranges (like Aptitude_Test_Score: 0-100) would dominate features with small ranges (like Internships: 0-5), biasing the model.

### Q21. What is a confusion matrix?
**A:** A confusion matrix is a 2×2 table showing: True Positives (correctly predicted Placed), True Negatives (correctly predicted Not Placed), False Positives (predicted Placed but actually Not Placed), False Negatives (predicted Not Placed but actually Placed).

### Q22. What is the bias-variance tradeoff?
**A:** Bias is the error from overly simplistic assumptions (underfitting). Variance is the error from being too sensitive to training data (overfitting). Random Forest achieves a good balance — low bias (complex individual trees) and low variance (averaging many trees).

### Q23. What is KMeans clustering?
**A:** KMeans is an unsupervised algorithm that partitions data into K groups by: (1) placing K random centroids, (2) assigning each point to the nearest centroid, (3) recalculating centroids as cluster means, (4) repeating until convergence. We use the Silhouette Score to find the optimal K.

### Q24. What is the Silhouette Score?
**A:** It measures how similar a data point is to its own cluster compared to other clusters. Range: -1 to +1. A high silhouette score means clusters are well-separated. We evaluate K=2 to K=10 and select the K with the highest average silhouette score.

### Q25. What is the difference between Gradient Boosting and Random Forest?
**A:** Random Forest builds trees in parallel (independently) and averages them. Gradient Boosting builds trees sequentially — each new tree corrects the errors of the previous tree. Gradient Boosting often achieves higher accuracy but is more prone to overfitting and slower to train.

---

## Category 3: Deep Learning / LLM (10 Questions)

### Q26. What is a Transformer?
**A:** A Transformer is a neural network architecture based on the self-attention mechanism. It processes all tokens in a sequence simultaneously (not sequentially like RNNs), making it highly parallelisable and effective for NLP tasks. BERT, GPT, and DistilBERT are all Transformer models.

### Q27. What is BERT?
**A:** BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that understands the context of a word based on ALL surrounding words (left and right). It was trained on Wikipedia and BookCorpus using Masked Language Modelling and Next Sentence Prediction.

### Q28. What is DistilBERT and why did we choose it?
**A:** DistilBERT is a smaller, faster version of BERT created through knowledge distillation. It has 66M parameters (vs BERT's 110M), is 60% faster, and retains 97% of BERT's performance. We chose it because it can run on a regular laptop CPU without needing a GPU.

### Q29. What is Transfer Learning?
**A:** Transfer Learning is the technique of taking a model pre-trained on a large general dataset and fine-tuning it on a smaller, task-specific dataset. Instead of training DistilBERT from scratch (which would need billions of tokens and weeks of compute), we only train the classification head on our placement data.

### Q30. What is fine-tuning?
**A:** Fine-tuning means continuing the training of a pre-trained model on new, task-specific data with a very small learning rate. The pre-trained weights are slightly adjusted (not frozen) while the new classification layer is trained from scratch. We fine-tune for 3 epochs with a learning rate of 2e-5.

### Q31. What is tokenisation?
**A:** Tokenisation is the process of breaking text into tokens (sub-word units) that the model can process. DistilBERT uses WordPiece tokenisation, which breaks unknown words into known sub-word pieces. For example, "internship" might become ["intern", "##ship"].

### Q32. What is the [CLS] token?
**A:** [CLS] is a special token prepended to every input in BERT/DistilBERT. After processing the entire sequence, the [CLS] token's embedding captures a summary representation of the entire input. This embedding is fed into the classification head for making predictions.

### Q33. Why did we generate synthetic resume text instead of using real resumes?
**A:** We don't have real resume texts paired with placement outcomes. Instead, we convert each row from cleaned_train.csv into a descriptive paragraph. This is a practical approach that teaches the model to associate profile descriptions with placement outcomes.

### Q34. What loss function does the LLM use?
**A:** CrossEntropyLoss, which is the standard loss for multi-class (and binary) classification. It measures how far the model's predicted probability distribution is from the true label.

### Q35. What is the attention mechanism?
**A:** Attention allows the model to focus on different parts of the input when making predictions. Self-attention computes a weighted sum of all input tokens, where the weights represent how much each token is relevant to every other token. This captures long-range dependencies in text.

---

## Category 4: Explainable AI (5 Questions)

### Q36. What is Explainable AI (XAI)?
**A:** XAI refers to techniques that make AI model predictions understandable to humans. Instead of treating the model as a "black box", XAI shows WHICH features contributed to a specific prediction and by HOW MUCH.

### Q37. What is SHAP?
**A:** SHAP (SHapley Additive exPlanations) is a method from game theory that assigns each feature an importance value for a particular prediction. It answers: "How much did each feature contribute to pushing the prediction above or below the average?"

### Q38. What is the difference between global and local explanations?
**A:** Global explanations show which features are important across ALL predictions (feature importance bar chart). Local explanations show which features drove a SPECIFIC individual prediction (SHAP waterfall for one student).

### Q39. How does TreeExplainer work?
**A:** TreeExplainer is a fast, exact SHAP algorithm specifically designed for tree-based models. It computes SHAP values by efficiently traversing all possible feature subsets through the tree structure, which is much faster than the general KernelExplainer.

### Q40. Why do we use a fallback chain for SHAP?
**A:** Different model types require different SHAP explainers. We try TreeExplainer first (fastest, for tree models), then LinearExplainer (for logistic regression), then KernelExplainer (universal fallback). This ensures the system never crashes regardless of which model is selected as best.

---

## Category 5: Data & Preprocessing (10 Questions)

### Q41. Why do we encode categorical variables?
**A:** ML models need numerical inputs. Categorical variables like "Gender" (Male/Female) must be converted to numbers (0/1). We use LabelEncoder which assigns a unique integer to each category.

### Q42. What is the IQR method for outlier detection?
**A:** IQR = Q3 - Q1 (interquartile range). Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are outliers. We cap (clip) them to these bounds rather than removing them, to preserve data volume.

### Q43. Why do we use StandardScaler instead of MinMaxScaler?
**A:** StandardScaler (z-score normalisation) is less sensitive to outliers than MinMaxScaler. It transforms data to mean=0 and std=1, which works better with gradient-based algorithms and distance-based models.

### Q44. What does the target variable look like?
**A:** The target is "Placement_Status" with values "Placed" and "Not Placed". We map "Placed" → 1 and "Not Placed" → 0 for model training.

### Q45. Why do we remove Student_ID from features?
**A:** Student_ID is a unique identifier with no predictive value. If included, a model might memorise specific IDs rather than learning general patterns.

### Q46. How do we handle missing values?
**A:** Numerical features: filled with the median (robust to outliers). Categorical features: filled with the mode (most frequent value).

### Q47. Why do we concatenate train and test data for preprocessing?
**A:** To ensure the LabelEncoder and StandardScaler see all possible values. If a category exists in test but not train, the encoder would crash. After fitting, we split back using original row counts.

### Q48. What is class imbalance?
**A:** When one class (e.g., "Not Placed") has significantly more samples than the other. Our dataset has ~64% Not Placed and ~36% Placed. We use stratified splits to maintain this ratio across train/test/CV folds.

### Q49. What are composite features?
**A:** New features created by combining multiple raw features. For example, AcademicScore = 0.7 × normalised(CGPA) - 0.3 × normalised(Backlogs). These capture higher-level patterns.

### Q50. Why normalise composite features to [0, 1]?
**A:** To ensure all composite features are on the same scale. Without normalisation, a feature calculated from 0-100 range columns would dominate one calculated from 0-10 range columns.

---

## Category 6: Practical & Code-Specific (10 Questions)

### Q51. How do you run the project?
**A:** (1) Navigate to PlacementIQ_AI/, (2) Activate venv: `source venv/Scripts/activate`, (3) Train models: `python src/predict.py --train`, (4) Train LLM: `python src/predict.py --train-llm`, (5) Predict: `python src/predict.py --interactive`.

### Q52. What is a virtual environment and why do we use one?
**A:** A virtual environment is an isolated Python installation that keeps project dependencies separate from the system Python. This prevents version conflicts between projects and ensures reproducibility.

### Q53. How does the company matching algorithm work?
**A:** It uses placement probability thresholds:
- >80%: Dream companies (Google, Microsoft)
- >60%: Aspirational (Flipkart, Adobe)
- >40%: Strong (TCS Digital, Mindtree)
- >20%: Mass recruiters (TCS, Infosys)

### Q54. How is the 4-week roadmap generated?
**A:** The recommendation engine identifies the student's weakest skill categories (below 60% score). It then assigns specific, actionable tasks for each week, prioritising the weakest area first.

### Q55. What is the ATS resume scorer?
**A:** It evaluates resumes against Applicant Tracking System criteria using keyword matching. It checks for technical skills (Python, Docker, etc.), action verbs (developed, implemented), essential sections (Education, Experience), and formatting keywords (LinkedIn, GitHub).

### Q56. What is joblib and why do we use it?
**A:** Joblib is a Python library for serialising (saving) Python objects to disk. We use it to save trained models, scalers, and encoders so we can load them later for predictions without retraining.

### Q57. How does the interactive mode work internally?
**A:** It uses Python's `input()` function to collect each feature value. The values are assembled into a dictionary, which is passed to `predict_single_student()`. This function applies feature engineering, encoding, scaling, prediction, clustering, SHAP, skill analysis, company matching, and report generation — all in sequence.

### Q58. What happens if a user enters an unknown category (e.g., Gender="Other")?
**A:** The LabelEncoder won't recognise it. The code handles this gracefully by assigning it a default value of 0 (fallback for unknown categories) rather than crashing.

### Q59. Can you explain the modular architecture?
**A:** Each concern is separated into its own Python file: data loading, preprocessing, feature engineering, training, evaluation, clustering, explanation, recommendations, company matching, resume scoring, interview scoring, and report generation. The `predict.py` orchestrator imports and calls these modules in sequence.

### Q60. What improvements would you suggest for future work?
**A:** (1) Use real resume text data instead of synthetically generated text, (2) Add a web-based frontend (Flask/Streamlit), (3) Incorporate more features like LinkedIn profiles and GitHub contributions, (4) Use a larger LLM (BERT-base or RoBERTa) with GPU training, (5) Add A/B testing to compare model versions, (6) Deploy as a REST API with authentication.

---

## Category 7: General AI/ML Questions (5 Questions)

### Q61. What is the difference between AI, ML, and Deep Learning?
**A:** AI is the broad field of making machines intelligent. ML is a subset of AI where machines learn from data. Deep Learning is a subset of ML using neural networks with many layers. Our project uses classical ML (Random Forest) AND deep learning (DistilBERT Transformer).

### Q62. What is the difference between a parameter and a hyperparameter?
**A:** Parameters are learned from data (e.g., tree split thresholds, neural network weights). Hyperparameters are set before training (e.g., n_estimators=200, learning_rate=0.1, epochs=3).

### Q63. What are the ethical considerations of placement prediction?
**A:** The model should not discriminate based on gender, age, or other protected attributes. We include a disclaimer that results are guidance only. The model should be regularly audited for bias, and no student should be denied opportunities based solely on AI predictions.

### Q64. What is model interpretability and why does it matter?
**A:** Interpretability means understanding WHY a model made a decision. In placement prediction, students need to know WHAT to improve. SHAP provides this — it shows whether CGPA helped or hurt their prediction, enabling actionable advice.

### Q65. How would you improve the model's AUC if it were lower than 0.90?
**A:** (1) Collect more training data, (2) Engineer more features, (3) Try ensemble methods (stacking multiple models), (4) Hyperparameter tuning using GridSearchCV or Optuna, (5) Address class imbalance with SMOTE, (6) Try more complex models like neural networks.

---

## Appendix: Core Mathematical Formulas 
*These are highly likely to be asked by an examiner during a Viva or presentation to test your core understanding.*

### 1. Feature Engineering Formulas (Custom Logic)
*   **Academic Score:**
    $$AcademicScore = \left( \frac{CGPA}{10} \times 0.6 \right) - \left( \frac{Backlogs}{5} \times 0.4 \right)$$
*   **Technical Score:**
    $$TechnicalScore = \frac{Coding\_Skills + Projects + \left(\frac{Aptitude\_Test\_Score}{10}\right)}{30}$$

### 2. Data Pre-Processing Formulas
*   **Standard Scaler (Z-Score Normalization):** Used to scale numerical features so they have a mean of 0 and standard deviation of 1.
    $$z = \frac{x - \mu}{\sigma}$$
    *(Where $x$ is the value, $\mu$ is the mean, and $\sigma$ is the standard deviation)*
*   **Min-Max Scaler:** Used to scale your engineered scores strictly between 0 and 1.
    $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
*   **Outlier Detection (Interquartile Range - IQR):** Used during data cleaning to find and cap outliers (the dots on the boxplots).
    $$IQR = Q3 - Q1$$
    $$Lower Bound = Q1 - 1.5 \times IQR$$
    $$Upper Bound = Q3 + 1.5 \times IQR$$

### 3. Model Evaluation Metrics
*   **Accuracy:** The percentage of total predictions the model got right.
    $$Accuracy = \frac{True Positives + True Negatives}{Total Predictions}$$
*   **Precision:** Out of all students the model *predicted* would be placed, how many actually were?
    $$Precision = \frac{True Positives}{True Positives + False Positives}$$
*   **Recall (Sensitivity):** Out of all students who were *actually* placed, how many did the model find?
    $$Recall = \frac{True Positives}{True Positives + False Negatives}$$
*   **F1-Score:** The harmonic mean of Precision and Recall (used because the dataset was slightly imbalanced).
    $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 4. Machine Learning Algorithms (Core Math)
*   **Logistic Regression:** Uses the Sigmoid function to output a probability between 0 and 1.
    $$\hat{y} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}$$
*   **Random Forest / Decision Tree:** Uses "Gini Impurity" to decide how to split the tree branches (lower Gini means a better, purer split).
    $$Gini = 1 - \sum (p_i)^2$$
    *(Where $p_i$ is the probability of an item belonging to a specific class)*
*   **K-Means Clustering (Student Segmentation):** Minimizes the "Inertia" (Within-Cluster Sum of Squares) to group similar students together.
    $$Inertia = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
