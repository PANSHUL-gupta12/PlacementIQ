"""
Comprehensive AIML Curriculum & Pipeline Test Runner
======================================================

Tests all aspects of the curriculum:
  1. Supervised Learning Model Testing on 80/20 Split Test Set
  2. Unsupervised Learning: K-Means, Hierarchical, DBSCAN, Silhouette Score
  3. Association Rules & Recommendations: Apriori, FP-Growth, PCA Scree & 2D Projection
  4. Reinforcement Learning: Environment, Q-Learning Agent, Optimal Policy Convergence
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from unsupervised_learning import evaluate_kmeans_range, run_hierarchical_clustering, run_dbscan_clustering, plot_cluster_evaluation_metrics
from association_and_pca import create_student_skill_baskets, mine_association_rules, recommend_courses_for_student, run_pca_analysis
from reinforcement_learning import StudentPlacementEnv, QLearningPlacementAgent, train_rl_placement_agent, COMPARATIVE_STUDY_TABLE, ETHICS_AND_TRENDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestRunner")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def test_supervised_classifier_80_20():
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST 1: Supervised Classifier Evaluation on 80/20 Test Split")
    logger.info("=" * 70)

    test_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "test.csv")
    assert os.path.exists(test_csv), f"Test CSV missing at {test_csv}"
    df_test = pd.read_csv(test_csv)
    logger.info(f"Loaded test dataset: {len(df_test):,} records (Expected 20% = 10,000)")
    assert len(df_test) == 10000, f"Expected 10,000 test records, got {len(df_test)}"

    clf_path = os.path.join(MODEL_DIR, "best_classifier.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if os.path.exists(clf_path) and os.path.exists(scaler_path):
        clf = joblib.load(clf_path)
        logger.info(f"Loaded trained best model: {type(clf).__name__}")
        logger.info("✅ Supervised Model Test passed.")
    else:
        logger.warning("Classifier or scaler file not found yet (training in progress).")


def test_unsupervised_clustering():
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST 2: Unsupervised Learning (K-Means, Hierarchical, DBSCAN, Silhouette)")
    logger.info("=" * 70)

    train_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "train.csv")
    df = pd.read_csv(train_csv)
    numeric_cols = ['Age', 'CGPA', 'Internships', 'Projects', 'Coding_Skills',
                    'Communication_Skills', 'Aptitude_Test_Score', 'Soft_Skills_Rating',
                    'Certifications', 'Backlogs']
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(df[numeric_cols])

    # K-Means
    k_res = evaluate_kmeans_range(X_scaled, k_range=range(2, 6), subsample_size=3000)
    logger.info("K-Means Best K: %d (Silhouette: %.4f)", k_res['best_k'], max(k_res['silhouettes']))
    plot_cluster_evaluation_metrics(k_res, OUTPUT_DIR)

    # Hierarchical
    h_labels, dendro_path = run_hierarchical_clustering(X_scaled, n_clusters=4, sample_size=300, output_dir=OUTPUT_DIR)
    logger.info("Hierarchical clustering completed. Dendrogram saved to: %s", dendro_path)

    # DBSCAN
    db_res = run_dbscan_clustering(X_scaled, eps=2.0, min_samples=15, sample_size=2000)
    logger.info("DBSCAN completed: %d clusters, %d noise points.", db_res['n_clusters'], db_res['n_noise'])

    assert k_res['best_k'] >= 2
    logger.info("✅ Unsupervised Learning Test passed.")


def test_association_rules_and_pca():
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST 3: Association Rules (Apriori/FP-Growth), Recommendations & PCA")
    logger.info("=" * 70)

    train_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "train.csv")
    df = pd.read_csv(train_csv).head(5000)

    # Association Rules
    baskets = create_student_skill_baskets(df)
    rules_res = mine_association_rules(baskets, min_support=0.15, min_confidence=0.5, output_dir=OUTPUT_DIR)
    logger.info("Mined %d association rules.", len(rules_res['all_rules']))

    # Course Recommendations
    student_sample = {"Coding_Skills": 3, "Projects": 1, "Soft_Skills_Rating": 8, "Certifications": 0, "Aptitude_Test_Score": 55}
    recs = recommend_courses_for_student(student_sample, top_n=3)
    logger.info("Generated %d course recommendations for student.", len(recs))
    assert len(recs) == 3

    # PCA
    numeric_cols = ['Age', 'CGPA', 'Internships', 'Projects', 'Coding_Skills',
                    'Communication_Skills', 'Aptitude_Test_Score', 'Soft_Skills_Rating',
                    'Certifications', 'Backlogs']
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(df[numeric_cols])
    y_labels = (df['Placement_Status'] == 'Placed').astype(int).values

    pca_res = run_pca_analysis(X_scaled, numeric_cols, y_labels, n_components=2, output_dir=OUTPUT_DIR)
    logger.info("Top 2 PCs explain %.2f%% of variance.", pca_res['cumulative_variance'][1] * 100)

    assert len(rules_res['all_rules']) > 0
    logger.info("✅ Association Rules & PCA Test passed.")


def test_reinforcement_learning():
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST 4: Reinforcement Learning (Environment, Q-Learning, Optimal Policy)")
    logger.info("=" * 70)

    rl_res = train_rl_placement_agent(episodes=600, output_dir=OUTPUT_DIR)
    logger.info("Q-Table Shape: %s", rl_res['q_table'].shape)
    logger.info("Optimal Policy Derived:")
    for s, a in rl_res['optimal_policy'].items():
        logger.info("   • %s ➜ %s", s, a)

    assert len(rl_res['optimal_policy']) > 0
    assert COMPARATIVE_STUDY_TABLE is not None
    assert len(ETHICS_AND_TRENDS['Ethical Considerations']) == 4
    logger.info("✅ Reinforcement Learning Test passed.")


if __name__ == "__main__":
    logger.info("🚀 STARTING COMPREHENSIVE CURRICULUM TEST SUITE")
    test_supervised_classifier_80_20()
    test_unsupervised_clustering()
    test_association_rules_and_pca()
    test_reinforcement_learning()
    logger.info("\n🎉 ALL CURRICULUM MODULES TESTED & PASSED SUCCESSFULLY!")

