"""
Association Rule Learning, Recommendation Systems & PCA
==========================================================

Covers Topics 25-27:
  1. Association Rule Learning & Frequent Itemset Mining
  2. Apriori Algorithm (Support, Confidence, Lift)
  3. FP-Growth Algorithm (Fast Frequent Itemset Mining via FP-Tree)
  4. Market Basket Analysis for Student Competency & Placements
  5. Recommendation Systems (Content-Based Cosine Similarity & Profile Matching)
  6. Dimensionality Reduction via Principal Component Analysis (PCA)
     - Eigenvalues & Eigenvectors
     - Explained Variance Ratio & Scree Plot
     - 2D/3D Feature Projection
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Market Basket Analysis Data Generation from Student Records
# ═══════════════════════════════════════════════════════════════════════════
def create_student_skill_baskets(df: pd.DataFrame) -> List[List[str]]:
    """Convert continuous and categorical student records into discrete skill/achievement baskets."""
    baskets = []
    for _, row in df.iterrows():
        basket = []
        # Academic tier
        cgpa = row.get("CGPA", 0)
        if cgpa >= 8.5:
            basket.append("CGPA_High(>=8.5)")
        elif cgpa >= 7.0:
            basket.append("CGPA_Mid(7.0-8.5)")
        else:
            basket.append("CGPA_Low(<7.0)")

        # Coding skills
        coding = row.get("Coding_Skills", 0)
        if coding >= 7:
            basket.append("Coding_Strong(>=7)")
        elif coding >= 4:
            basket.append("Coding_Average(4-6)")
        else:
            basket.append("Coding_Novice(<4)")

        # Internships & Projects
        if row.get("Internships", 0) >= 1:
            basket.append("Has_Internship")
        if row.get("Projects", 0) >= 3:
            basket.append("Projects_Rich(>=3)")
        if row.get("Certifications", 0) >= 2:
            basket.append("Multi_Certified")
        if row.get("Backlogs", 0) == 0:
            basket.append("Zero_Backlogs")
        if row.get("Soft_Skills_Rating", 0) >= 7:
            basket.append("High_Soft_Skills")

        # Target outcome
        status = row.get("Placement_Status", "")
        if str(status) in ["Placed", "1", 1]:
            basket.append("Placement_Success")
        else:
            basket.append("Placement_Pending")

        baskets.append(basket)
    return baskets


# ═══════════════════════════════════════════════════════════════════════════
# 2. Frequent Itemset Mining & Association Rules (Apriori vs FP-Growth)
# ═══════════════════════════════════════════════════════════════════════════
def mine_association_rules(
    baskets: List[List[str]],
    min_support: float = 0.15,
    min_confidence: float = 0.50,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Execute both Apriori and FP-Growth algorithms and generate actionable association rules."""
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 1. Apriori
    frequent_itemsets_apriori = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # 2. FP-Growth (Faster, avoids candidate generation)
    frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    # 3. Association Rules
    rules = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

    # Filter rules pointing to Placement_Success
    placement_rules = rules[rules["consequents"].apply(lambda s: "Placement_Success" in s)].copy()

    os.makedirs(output_dir, exist_ok=True)
    rules_path = os.path.join(output_dir, "placement_association_rules.csv")
    rules.to_csv(rules_path, index=False)
    logger.info("Mined %d frequent itemsets and %d association rules.", len(frequent_itemsets_fpgrowth), len(rules))

    return {
        "encoded_df": df_encoded,
        "apriori_itemsets": frequent_itemsets_apriori,
        "fpgrowth_itemsets": frequent_itemsets_fpgrowth,
        "all_rules": rules,
        "placement_rules": placement_rules,
        "rules_csv": rules_path,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Recommendation System (Content-Based Skill Matching)
# ═══════════════════════════════════════════════════════════════════════════
COURSE_CATALOG = [
    {"Course_ID": "C01", "Title": "Data Structures & Competitive Coding", "Category": "Coding", "Focus": "DSA, Algorithms, LeetCode", "Level": "Advanced"},
    {"Course_ID": "C02", "Title": "Full-Stack Web Engineering with React & Node", "Category": "Projects", "Focus": "React, APIs, Databases, Deployment", "Level": "Intermediate"},
    {"Course_ID": "C03", "Title": "Enterprise Cloud Architecture (AWS / GCP)", "Category": "Certifications", "Focus": "Cloud computing, Containers, DevOps", "Level": "Intermediate"},
    {"Course_ID": "C04", "Title": "Executive Business Communication & GD Mastery", "Category": "Soft Skills", "Focus": "Group discussion, Mock interviews, Presentation", "Level": "All"},
    {"Course_ID": "C05", "Title": "Machine Learning & Deep Learning Foundations", "Category": "AI/ML", "Focus": "Scikit-Learn, PyTorch, Model Deployment", "Level": "Advanced"},
    {"Course_ID": "C06", "Title": "Quantitative Aptitude & Logical Reasoning Sprint", "Category": "Aptitude", "Focus": "Probability, Algebra, Puzzles, Speed Math", "Level": "All"},
]


def recommend_courses_for_student(
    student_vector: Dict[str, float],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Recommend skill-boosting courses based on student profile weaknesses."""
    priorities = []
    if student_vector.get("Coding_Skills", 5) < 6:
        priorities.append("Coding")
    if student_vector.get("Projects", 2) < 3:
        priorities.append("Projects")
    if student_vector.get("Soft_Skills_Rating", 5) < 6 or student_vector.get("Communication_Skills", 5) < 6:
        priorities.append("Soft Skills")
    if student_vector.get("Aptitude_Test_Score", 60) < 65:
        priorities.append("Aptitude")
    if student_vector.get("Certifications", 0) < 2:
        priorities.append("Certifications")

    recommendations = []
    for p in priorities:
        for c in COURSE_CATALOG:
            if c["Category"] == p and c not in recommendations:
                recommendations.append(c)
                if len(recommendations) >= top_n:
                    break
        if len(recommendations) >= top_n:
            break

    # Fallback to general high-value courses
    for c in COURSE_CATALOG:
        if len(recommendations) >= top_n:
            break
        if c not in recommendations:
            recommendations.append(c)

    return recommendations[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dimensionality Reduction: Principal Component Analysis (PCA)
# ═══════════════════════════════════════════════════════════════════════════
def run_pca_analysis(
    X_scaled: np.ndarray,
    feature_names: List[str],
    y_labels: Optional[np.ndarray] = None,
    n_components: int = 2,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Execute PCA, compute explained variance, and generate scree plot + 2D projection."""
    # Full PCA for explained variance / Scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    # 2D PCA for visualization
    pca_2d = PCA(n_components=n_components)
    X_pca = pca_2d.fit_transform(X_scaled)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Scree Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    components = range(1, len(var_ratio) + 1)
    ax1.bar(components, var_ratio, alpha=0.6, color="royalblue", label="Individual Variance")
    ax1.plot(components, cum_var_ratio, "r-o", linewidth=2, label="Cumulative Variance")
    ax1.axhline(0.80, color="green", linestyle="--", label="80% Threshold")
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Scree Plot & Cumulative Variance", fontweight="bold")
    ax1.legend()

    # 2. 2D Projection
    subsample_n = min(2000, len(X_pca))
    np.random.seed(42)
    sub_idx = np.random.choice(len(X_pca), subsample_n, replace=False)
    scatter = ax2.scatter(
        X_pca[sub_idx, 0],
        X_pca[sub_idx, 1],
        c=y_labels[sub_idx] if y_labels is not None else None,
        cmap="coolwarm" if y_labels is not None else None,
        alpha=0.6,
        edgecolors="none",
    )
    ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)")
    ax2.set_title("Student Feature Space Projected onto Top 2 PCs", fontweight="bold")
    if y_labels is not None:
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label("Placement Outcome (0=Not Placed, 1=Placed)")

    fig.tight_layout()
    pca_plot_path = os.path.join(output_dir, "pca_variance_and_projection.png")
    fig.savefig(pca_plot_path, dpi=150)
    plt.close(fig)

    # Component loadings (Which original features drive PC1 and PC2)
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names,
    )
    loadings.to_csv(os.path.join(output_dir, "pca_feature_loadings.csv"))
    logger.info("PCA Scree plot and 2D projection saved to %s", pca_plot_path)

    return {
        "pca_full": pca_full,
        "pca_2d": pca_2d,
        "X_pca": X_pca,
        "explained_variance_ratio": var_ratio,
        "cumulative_variance": cum_var_ratio,
        "loadings": loadings,
        "plot_path": pca_plot_path,
    }

