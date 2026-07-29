"""
Company Matching — Rule-Based Company Recommendation Engine
=============================================================

Recommends companies based on a student's placement probability,
technical skills, and profile. Companies are grouped into tiers:
  - Tier 1 (Dream): Google, Microsoft, Amazon, Meta, Apple
  - Tier 2 (Aspirational): Flipkart, Uber, Goldman Sachs, Adobe, PayPal
  - Tier 3 (Strong): TCS Digital, Infosys PP, Wipro, Cognizant, Capgemini
  - Tier 4 (Mass): TCS, Infosys, Accenture, Wipro, HCL

⚠️ This module NEVER claims guaranteed placement.
"""

import logging
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Company Tier Definitions
# ═══════════════════════════════════════════════════════════════════════════
COMPANY_TIERS: list[dict] = [
    {
        "tier": 1,
        "label": "Dream Companies",
        "min_probability": 0.85,
        "min_coding": 0.75,
        "min_cgpa": 0.70,
        "companies": ["Google", "Microsoft", "Amazon", "Meta", "Apple"],
        "note": "Highly competitive — strong DSA and system design required",
    },
    {
        "tier": 2,
        "label": "Aspirational Companies",
        "min_probability": 0.70,
        "min_coding": 0.60,
        "min_cgpa": 0.60,
        "companies": ["Flipkart", "Uber", "Goldman Sachs", "Adobe", "PayPal", "Salesforce"],
        "note": "Competitive — strong coding and problem-solving required",
    },
    {
        "tier": 3,
        "label": "Strong Companies",
        "min_probability": 0.55,
        "min_coding": 0.45,
        "min_cgpa": 0.50,
        "companies": ["TCS Digital", "Infosys Power Programmer", "Mindtree", "Cognizant", "Capgemini"],
        "note": "Good profiles needed — balanced academics and technical skills",
    },
    {
        "tier": 4,
        "label": "Mass Recruiters",
        "min_probability": 0.35,
        "min_coding": 0.25,
        "min_cgpa": 0.40,
        "companies": ["TCS", "Infosys", "Accenture", "Wipro", "HCL", "Tech Mahindra"],
        "note": "Accessible — aptitude and basic coding required",
    },
]

# Minimum threshold below which we recommend "Focus on improvement first"
ABSOLUTE_MIN_PROBABILITY: float = 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Matching Logic
# ═══════════════════════════════════════════════════════════════════════════
def match_companies(
    placement_probability: float,
    student_data: dict[str, float],
) -> dict:
    """Match a student to company tiers based on their profile.

    Args:
        placement_probability: Predicted placement probability (0–1).
        student_data: Dict of feature_name → normalised value.

    Returns:
        Dict with ``eligible``, ``stretch``, ``needs_improvement``, and
        ``recommendation_text``.
    """
    coding = student_data.get("CodingScore", student_data.get("CodingIndex", 0.0))
    cgpa = student_data.get("CGPA", student_data.get("AcademicScore", 0.0))

    eligible: list[dict] = []
    stretch: list[dict] = []
    needs_improvement: list[dict] = []

    for tier in COMPANY_TIERS:
        meets_prob = placement_probability >= tier["min_probability"]
        meets_coding = coding >= tier["min_coding"]
        meets_cgpa = cgpa >= tier["min_cgpa"]

        entry = {
            "tier": tier["tier"],
            "label": tier["label"],
            "companies": tier["companies"],
            "note": tier["note"],
        }

        if meets_prob and meets_coding and meets_cgpa:
            eligible.append(entry)
        elif meets_prob or (meets_coding and meets_cgpa):
            stretch.append(entry)
        else:
            needs_improvement.append(entry)

    result = {
        "eligible": eligible,
        "stretch": stretch,
        "needs_improvement": needs_improvement,
        "placement_probability": round(placement_probability, 4),
    }

    logger.info("Company matching — Eligible tiers: %d, Stretch: %d, Needs improvement: %d",
                len(eligible), len(stretch), len(needs_improvement))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Format as Text
# ═══════════════════════════════════════════════════════════════════════════
def format_company_text(matching: dict) -> str:
    """Format company recommendations as human-readable text.

    Args:
        matching: Output from ``match_companies()``.

    Returns:
        Formatted string.
    """
    lines = [
        "🏢 Company Recommendations",
        "=" * 40,
        "",
        "⚠️  Disclaimer: These are potential targets based on profile analysis.",
        "   No placement is ever guaranteed.",
        "",
    ]

    if matching["eligible"]:
        lines.append("✅ Eligible (Profile matches well):")
        for tier in matching["eligible"]:
            companies_str = ", ".join(tier["companies"])
            lines.append(f"   [{tier['label']}] {companies_str}")
            lines.append(f"   💡 {tier['note']}")
            lines.append("")

    if matching["stretch"]:
        lines.append("🔶 Stretch (Profile partially matches):")
        for tier in matching["stretch"]:
            companies_str = ", ".join(tier["companies"])
            lines.append(f"   [{tier['label']}] {companies_str}")
            lines.append(f"   💡 {tier['note']}")
            lines.append("")

    if matching["needs_improvement"]:
        lines.append("🔴 Needs Improvement:")
        for tier in matching["needs_improvement"]:
            companies_str = ", ".join(tier["companies"])
            lines.append(f"   [{tier['label']}] {companies_str}")
            lines.append(f"   💡 Focus on improving to reach this tier")
            lines.append("")

    if matching["placement_probability"] < ABSOLUTE_MIN_PROBABILITY:
        lines.append("⚠️  Your profile needs significant improvement.")
        lines.append("   Focus on building core skills before targeting companies.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample_data = {
        "CodingScore": 0.72,
        "CGPA": 0.75,
        "CodingIndex": 0.70,
        "AcademicScore": 0.68,
    }
    result = match_companies(0.78, sample_data)
    print(format_company_text(result))
