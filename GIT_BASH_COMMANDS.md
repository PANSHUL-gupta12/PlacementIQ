# PlacementIQ AI — Master Git Bash & Terminal Commands Reference

All commands below are formatted specifically for **Git Bash (MinGW64)** on Windows.

---

## 1. Environment Setup & Navigation (Run First)

```bash
# Navigate to the project root directory
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI

# Verify you are in the correct directory
pwd

# Activate the virtual environment
source venv/Scripts/activate

# Verify Python executable (must point to venv/Scripts/python)
which python
python --version
```

---

## 2. Verify All Modules via Command Line (From Screenshot)

```bash
# 1. Run Complete Automated Syllabus Test Suite
python test_full_syllabus_suite.py

# 2. Run Continuous Package Regression Training
python src/train_regressor.py

# 3. Run K-Fold Cross-Validation & Learning Curve Diagnostics
python src/model_validation.py

# 4. Run Unsupervised K-Means, Dendrogram & PCA
python src/unsupervised_learning.py

# 5. Run Apriori Association Rule Miner
python src/association_and_pca.py

# 6. Run Q-Learning Reinforcement Learning Simulator
python src/reinforcement_learning.py

# 7. Run Interactive Single-Student Placement Predictor
python src/predict.py --interactive
```

---

## 3. Core Training & Model Pipeline Commands

```bash
# Train the primary 5-model classification suite (Random Forest, Gradient Boosting, etc.)
python src/train_classifier.py

# Fine-tune the DistilBERT NLP resume classifier (Optional Deep Learning Path)
python src/llm_resume_classifier.py

# Run all modular curriculum checks
python test_all_curriculum_modules.py
```

---

## 4. Inference, Batch Predictions & Reports

```bash
# Run interactive CLI predictor (prompts for CGPA, Coding, Projects, etc.)
python src/predict.py --interactive

# Run batch prediction on unseen test records (750 samples)
python src/predict.py --input dataset/test.csv --output outputs/batch_predictions.csv

# Generate institutional summary report text
python src/report_generator.py
```

---

## 5. Jupyter Notebooks via Git Bash

```bash
# Launch interactive Jupyter Lab in default browser
jupyter lab

# Or launch classic Jupyter Notebook server
jupyter notebook

# Execute notebooks headlessly (in-place execution without opening browser)
jupyter nbconvert --to notebook --execute 01_Phase1_Data_Preprocessing_and_EDA.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_Phase2_Supervised_Regression.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_Phase3_Supervised_Classification.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_Phase4_Model_Evaluation_and_Validation.ipynb --inplace
jupyter nbconvert --to notebook --execute 05_Phase5_Unsupervised_Clustering.ipynb --inplace
jupyter nbconvert --to notebook --execute 06_Phase6_Association_Rules_and_Recommendations.ipynb --inplace
jupyter nbconvert --to notebook --execute 07_Phase7_Reinforcement_Learning_and_Emerging_Trends.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_Comprehensive_AIML_Curriculum.ipynb --inplace
```

---

## 6. Git & GitHub Workflow Commands

```bash
# Fix dubious ownership exception on secondary drive (run once)
git config --global --add safe.directory D:/01_Projects/01_AimlProject/PlacementIQ_AI

# Configure Git user identity
git config user.name "Panshul"
git config user.email "panshul0739.becse24@chitkara.edu.in"

# Check working tree status
git status

# Stage all changes and new files
git add .

# Commit changes
git commit -m "Update CE-2 curriculum documentation, commands, and results"

# Push to GitHub main branch
git push origin main

# View recent commit history
git log --oneline -n 5

# Check configured remote repository
git remote -v
```

---

## 7. Inspect Outputs Directly in Git Bash

```bash
# View continuous package regression evaluation metrics
cat outputs/regression_metrics_comparison.csv

# View 5-model classification benchmark results
cat outputs/model_comparison.csv

# View top 10 mined association rules
head -n 11 outputs/placement_association_rules.csv

# View optimal Reinforcement Learning career policy
cat outputs/rl_optimal_policy.csv

# View PCA feature loadings matrix
cat outputs/pca_feature_loadings.csv

# View generated institutional summary report
cat outputs/placement_report.txt

# List all generated comparison PNG graphs
ls -lh outputs/*.png

# List all serialized joblib model files
ls -lh models/*.joblib
```

---

## 8. Common Troubleshooting in Git Bash

```bash
# If 'source venv/Scripts/activate' says file not found:
# Verify your directory first:
pwd
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI
source venv/Scripts/activate

# If pip packages fail to import:
python -m pip install -r requirements.txt

# If Git complains about CRLF vs LF line endings:
git config --global core.autocrlf true

# If Git refuses to push or requires safe directory:
git config --global --add safe.directory D:/01_Projects/01_AimlProject/PlacementIQ_AI
```

---

## 9. Master One-Liner Sequence (For Live Viva / Demo)

```bash
# Complete 30-second live demonstration:
cd /d/01_Projects/01_AimlProject/PlacementIQ_AI && source venv/Scripts/activate && python test_full_syllabus_suite.py && python src/predict.py --interactive
```
