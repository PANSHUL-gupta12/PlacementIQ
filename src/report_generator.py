"""
Report Generator — Final PlacementIQ AI Report
=================================================

Aggregates all module outputs into a comprehensive, formatted
placement report. Can output as:
  - Console text
  - Saved text file

The report includes:
  - Placement Probability & Readiness Level
  - Student Cluster
  - Strengths & Weaknesses
  - Company Recommendations
  - 4-Week Roadmap
  - Resume Analysis Summary (if available)
  - Interview Evaluation Summary (if available)
"""

import os
import logging
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Readiness Level Classification
# ═══════════════════════════════════════════════════════════════════════════
READINESS_LEVELS: list[tuple[float, str]] = [
    (0.85, "Highly Competitive"),
    (0.65, "Placement Ready"),
    (0.40, "Improving"),
    (0.00, "Beginner"),
]


def classify_readiness(probability: float) -> str:
    """Classify a placement probability into a readiness level.

    Args:
        probability: Placement probability (0–1).

    Returns:
        Readiness level string.
    """
    for threshold, level in READINESS_LEVELS:
        if probability >= threshold:
            return level
    return "Beginner"


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_report(
    probability: float,
    cluster_name: str,
    skill_analysis: dict,
    roadmap: list[dict],
    company_matching: dict,
    explanation: dict,
    resume_analysis: Optional[dict] = None,
    interview_result: Optional[dict] = None,
    student_name: str = "Student",
    predicted_salary: Optional[float] = None,
) -> str:
    """Generate the final PlacementIQ AI report.

    Args:
        probability: Placement probability (0–1).
        cluster_name: Student's cluster name.
        skill_analysis: Output from recommendation_engine.
        roadmap: 4-week roadmap from recommendation_engine.
        company_matching: Output from company_matching.
        explanation: Output from explain_prediction.
        resume_analysis: Output from resume_scoring (optional).
        interview_result: Output from interview_scoring (optional).
        student_name: Student identifier.

    Returns:
        Formatted report string.
    """
    readiness = classify_readiness(probability)
    prob_pct = round(probability * 100, 1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Header ──────────────────────────────────────────────────────────
    lines = [
        "",
        "╔" + "═" * 58 + "╗",
        "║" + "PLACEMENTIQ AI — PLACEMENT READINESS REPORT".center(58) + "║",
        "╚" + "═" * 58 + "╝",
        "",
        f"  📅 Generated: {timestamp}",
        f"  👤 Student:   {student_name}",
        "",
        "─" * 60,
    ]

    # ── Placement Prediction ────────────────────────────────────────────
    pred_lines = [
        "",
        "  📊 PLACEMENT PREDICTION & PACKAGE ESTIMATION",
        "  " + "─" * 40,
        f"  Placement Probability : {prob_pct}%",
        f"  Readiness Level       : {readiness}",
        f"  Student Cluster       : {cluster_name}",
    ]
    if predicted_salary is not None:
        pred_lines.append(f"  Estimated Package/CTC : {predicted_salary:.2f} LPA")
    pred_lines.append("")
    lines.extend(pred_lines)

    # ── Explanation ─────────────────────────────────────────────────────
    if explanation.get("positive_factors") or explanation.get("negative_factors"):
        lines.extend([
            "─" * 60,
            "",
            "  🔍 PREDICTION EXPLANATION",
            "  " + "─" * 40,
        ])
        if explanation.get("positive_factors"):
            lines.append("  Factors increasing placement chance:")
            for feat, val in explanation["positive_factors"][:5]:
                lines.append(f"    ✔ {feat}")
        if explanation.get("negative_factors"):
            lines.append("  Factors reducing placement chance:")
            for feat, val in explanation["negative_factors"][:5]:
                lines.append(f"    ✖ {feat}")
        lines.append("")

    # ── Strengths & Weaknesses ──────────────────────────────────────────
    lines.extend([
        "─" * 60,
        "",
        "  💪 STRENGTHS & WEAKNESSES",
        "  " + "─" * 40,
    ])

    if skill_analysis.get("strengths"):
        lines.append("  Strengths:")
        for s in skill_analysis["strengths"]:
            lines.append(f"    ✔ {s['category']} ({s['score']:.0%})")

    if skill_analysis.get("weaknesses"):
        lines.append("  Weaknesses:")
        for w in skill_analysis["weaknesses"]:
            lines.append(f"    ✖ {w['category']} ({w['score']:.0%})")
    lines.append("")

    # ── Company Recommendations ─────────────────────────────────────────
    lines.extend([
        "─" * 60,
        "",
        "  🏢 RECOMMENDED COMPANIES",
        "  " + "─" * 40,
    ])
    
    if company_matching.get("placement_probability", 0) >= 0.95:
        lines.append("  🏆 OUTSTANDING PROFILE: Placement is practically guaranteed!")
    else:
        lines.append("  ℹ️  Disclaimer: These are suggested targets. Placement is not guaranteed.")
        
    lines.append("")

    if company_matching.get("eligible"):
        lines.append("  ✅ Eligible:")
        for tier in company_matching["eligible"]:
            companies = ", ".join(tier["companies"][:4])
            lines.append(f"    ✔ [{tier['label']}] {companies}")

    if company_matching.get("stretch"):
        lines.append("  🔶 Stretch Targets:")
        for tier in company_matching["stretch"]:
            companies = ", ".join(tier["companies"][:4])
            lines.append(f"    → [{tier['label']}] {companies}")
            
    if not company_matching.get("eligible") and not company_matching.get("stretch"):
        lines.append("  🔴 Needs Improvement:")
        lines.append("    No company tiers currently match your profile.")
        lines.append("    Focus heavily on the roadmap below to improve your chances.")

    lines.append("")

    # ── Roadmap ─────────────────────────────────────────────────────────
    lines.extend([
        "─" * 60,
        "",
        "  🗺️  4-WEEK IMPROVEMENT ROADMAP",
        "  " + "─" * 40,
    ])

    for week in roadmap:
        lines.append(f"\n  📅 Week {week['week']}: {week['title']}")
        for task in week["tasks"][:3]:
            lines.append(f"    • {task}")
    lines.append("")

    # ── Resume Analysis (optional) ──────────────────────────────────────
    if resume_analysis:
        lines.extend([
            "─" * 60,
            "",
            "  📄 RESUME ANALYSIS",
            "  " + "─" * 40,
            f"  ATS Score: {resume_analysis.get('ats_score', 'N/A')}%",
        ])
        if resume_analysis.get("suggestions"):
            for sug in resume_analysis["suggestions"][:3]:
                lines.append(f"    → {sug}")
        lines.append("")

    # ── Interview Evaluation (optional) ─────────────────────────────────
    if interview_result:
        lines.extend([
            "─" * 60,
            "",
            "  🎤 INTERVIEW READINESS",
            "  " + "─" * 40,
            f"  Score: {interview_result.get('overall_score', 'N/A')}/100",
            f"  Readiness: {interview_result.get('readiness_level', 'N/A')}",
        ])
        if interview_result.get("weak_areas"):
            areas = ", ".join(a.replace("_", " ").title() for a in interview_result["weak_areas"])
            lines.append(f"  Weak Areas: {areas}")
        lines.append("")

    # ── Footer ──────────────────────────────────────────────────────────
    lines.extend([
        "─" * 60,
        "",
        "  ℹ️  This report is generated by PlacementIQ AI and is for",
        "     guidance purposes only. Results are based on statistical",
        "     models and do not guarantee placement outcomes.",
        "",
        "╔" + "═" * 58 + "╗",
        "║" + "END OF REPORT".center(58) + "║",
        "╚" + "═" * 58 + "╝",
        "",
    ])

    report = "\n".join(lines)
    logger.info("Report generated (%d lines)", len(lines))
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Save Report
# ═══════════════════════════════════════════════════════════════════════════
def save_report(
    report: str,
    output_dir: str,
    filename: str = "placement_report.txt",
) -> str:
    """Save the report to a text file.

    Args:
        report: The generated report string.
        output_dir: Directory to save.
        filename: Output filename.

    Returns:
        Full path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Quick demo with sample data
    report = generate_report(
        probability=0.89,
        cluster_name="Strong Coders",
        skill_analysis={
            "strengths": [
                {"category": "Technical", "score": 0.82},
                {"category": "Projects", "score": 0.75},
            ],
            "weaknesses": [
                {"category": "Communication", "score": 0.35},
                {"category": "Experience", "score": 0.28},
            ],
        },
        roadmap=[
            {"week": 1, "title": "Communication & Soft Skills", "tasks": ["Practice GD", "Record answers"]},
            {"week": 2, "title": "Experience Building", "tasks": ["Apply internships", "Get certified"]},
            {"week": 3, "title": "Coding Practice", "tasks": ["30 DSA problems", "System design"]},
            {"week": 4, "title": "Interview Prep", "tasks": ["Mock interviews", "Company research"]},
        ],
        company_matching={
            "eligible": [{"label": "Mass Recruiters", "companies": ["TCS", "Infosys", "Accenture"]}],
            "stretch": [{"label": "Strong Companies", "companies": ["Cognizant", "Capgemini"]}],
        },
        explanation={
            "positive_factors": [("CodingScore", 0.12), ("CGPA", 0.08), ("Projects", 0.06)],
            "negative_factors": [("Communication", -0.09), ("Internship", -0.07)],
        },
    )
    print(report)
