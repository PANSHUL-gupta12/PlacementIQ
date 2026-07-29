"""
Recommendation Engine — Skill Gap Analysis & Personalized Roadmap
===================================================================

Analyses student strengths and weaknesses across six categories:
  - Technical, Academic, Communication, Experience, Leadership, Projects

Generates:
  1. Skill gap report with improvement suggestions
  2. Personalized 4-week roadmap based on identified weaknesses
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Skill Category Definitions
# ═══════════════════════════════════════════════════════════════════════════
SKILL_CATEGORIES: dict[str, dict] = {
    "Technical": {
        "features": ["Coding_Skills", "Aptitude_Test_Score", "CodingIndex", "TechnicalScore"],
        "threshold": 0.55,
        "suggestions": [
            "Practice 20–30 DSA problems on LeetCode/HackerRank weekly",
            "Complete at least one competitive programming contest per month",
            "Study core CS subjects: OS, DBMS, Networking, OOP",
            "Work through a Data Structures & Algorithms course",
            "Solve aptitude and logical reasoning problems daily",
        ],
    },
    "Academic": {
        "features": ["CGPA", "Backlogs", "AcademicScore"],
        "threshold": 0.55,
        "suggestions": [
            "Maintain CGPA above 7.0 — attend classes regularly",
            "Clear any existing backlogs as soon as possible",
            "Attend workshops and academic seminars",
            "Form study groups for difficult subjects",
            "Use online courses (NPTEL, Coursera) to supplement learning",
        ],
    },
    "Communication": {
        "features": ["Communication_Skills", "CommunicationScore"],
        "threshold": 0.50,
        "suggestions": [
            "Join a public speaking club (Toastmasters, college club)",
            "Practice group discussions with peers regularly",
            "Read English newspapers and write daily summaries",
            "Watch TED talks and practice summarising them",
            "Participate in debates and extempore competitions",
        ],
    },
    "Experience": {
        "features": ["Internships", "Certifications", "ExperienceScore"],
        "threshold": 0.40,
        "suggestions": [
            "Apply for summer internships — even short-term ones add value",
            "Earn certifications (AWS, Google Cloud, Python, SQL)",
            "Volunteer for college technical fests and events",
            "Seek remote internships on Internshala, LinkedIn",
            "Shadow industry professionals for real-world exposure",
        ],
    },
    "Leadership": {
        "features": ["Soft_Skills_Rating"],
        "threshold": 0.45,
        "suggestions": [
            "Take leadership roles in college clubs or committees",
            "Organise technical events or coding competitions",
            "Mentor junior students in your domain",
            "Participate in community service projects",
            "Lead a team project or open-source contribution",
        ],
    },
    "Projects": {
        "features": ["Projects", "ProjectScore"],
        "threshold": 0.45,
        "suggestions": [
            "Build at least 2–3 real-world projects with deployment",
            "Contribute to open-source projects on GitHub",
            "Participate in hackathons (Smart India, HackWithInfy, etc.)",
            "Create a portfolio website showcasing your work",
            "Document all projects with READMEs and demos",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Skill Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════
def analyse_skill_gaps(
    student_data: dict[str, float],
) -> dict:
    """Analyse strengths and weaknesses for a student.

    Args:
        student_data: Dict of feature_name → value (normalised 0–1 preferred).

    Returns:
        Dict with ``strengths``, ``weaknesses``, and ``details`` per category.
    """
    strengths: list[dict] = []
    weaknesses: list[dict] = []
    details: dict[str, dict] = {}

    for category, spec in SKILL_CATEGORIES.items():
        # Get available feature values
        available_vals = [
            student_data.get(f, None) for f in spec["features"]
        ]
        valid_vals = [v for v in available_vals if v is not None]

        if not valid_vals:
            continue

        avg_score = float(np.mean(valid_vals))
        threshold = spec["threshold"]

        entry = {
            "category": category,
            "score": round(avg_score, 4),
            "threshold": threshold,
        }

        if avg_score >= threshold:
            entry["status"] = "strength"
            entry["suggestions"] = []
            strengths.append(entry)
        else:
            entry["status"] = "weakness"
            entry["suggestions"] = spec["suggestions"][:3]  # Top 3
            weaknesses.append(entry)

        details[category] = entry

    logger.info("Strengths: %s", [s["category"] for s in strengths])
    logger.info("Weaknesses: %s", [w["category"] for w in weaknesses])

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Personalized Roadmap
# ═══════════════════════════════════════════════════════════════════════════
# Week themes mapped to weakness categories
WEEK_THEMES: dict[str, dict] = {
    "Technical": {
        "week": 1,
        "title": "Coding & Technical Skills",
        "tasks": [
            "Complete 30 DSA problems (Arrays, Strings, Linked Lists)",
            "Revise core CS: OS process scheduling, DBMS normalisation",
            "Solve 10 aptitude/logical reasoning problems daily",
            "Set up a LeetCode/HackerRank profile and track progress",
        ],
    },
    "Projects": {
        "week": 2,
        "title": "Projects & Portfolio",
        "tasks": [
            "Build one full-stack mini project (todo app, blog, etc.)",
            "Push code to GitHub with proper README and documentation",
            "Start contributing to an open-source project",
            "Register for an upcoming hackathon",
        ],
    },
    "Communication": {
        "week": 3,
        "title": "Communication & Soft Skills",
        "tasks": [
            "Give a 5-minute presentation to friends/family daily",
            "Record yourself answering common HR questions",
            "Write a LinkedIn article or blog post",
            "Practice group discussion with peers (30 min)",
        ],
    },
    "Experience": {
        "week": 4,
        "title": "Interview Preparation & Experience",
        "tasks": [
            "Apply to 5 internships or entry-level roles",
            "Complete one online certification (Coursera, Udemy)",
            "Do 3 mock interviews (use Pramp, Interviewing.io)",
            "Prepare STAR-method answers for behavioural questions",
        ],
    },
}

# Fallback themes for categories without dedicated weeks
FALLBACK_TASKS: dict[str, list[str]] = {
    "Academic": [
        "Create a study plan for weak subjects",
        "Review previous semester exams",
        "Attend extra classes or tutorials",
    ],
    "Leadership": [
        "Volunteer to lead a college event",
        "Mentor a junior on a project",
        "Join a student committee",
    ],
}


def generate_roadmap(
    weaknesses: list[dict],
    strengths: list[dict],
) -> list[dict]:
    """Generate a personalized 4-week improvement roadmap.

    Weeks are allocated based on the student's identified weaknesses.
    If fewer than 4 weaknesses, remaining weeks focus on general growth.

    Args:
        weaknesses: List of weakness dicts from ``analyse_skill_gaps()``.
        strengths: List of strength dicts.

    Returns:
        List of 4 dicts, one per week, with ``week``, ``title``, ``tasks``.
    """
    roadmap: list[dict] = []
    used_categories: set[str] = set()

    # Allocate weeks to weaknesses first
    for weakness in weaknesses[:4]:
        cat = weakness["category"]
        used_categories.add(cat)

        if cat in WEEK_THEMES:
            week_plan = {
                "week": len(roadmap) + 1,
                "title": WEEK_THEMES[cat]["title"],
                "tasks": WEEK_THEMES[cat]["tasks"],
            }
        elif cat in FALLBACK_TASKS:
            week_plan = {
                "week": len(roadmap) + 1,
                "title": f"Improve {cat}",
                "tasks": FALLBACK_TASKS[cat] + weakness.get("suggestions", []),
            }
        else:
            week_plan = {
                "week": len(roadmap) + 1,
                "title": f"Improve {cat}",
                "tasks": weakness.get("suggestions", ["Focus on improvement"]),
            }
        roadmap.append(week_plan)

    # Fill remaining weeks with unused themes
    remaining_themes = [cat for cat in WEEK_THEMES if cat not in used_categories]
    for cat in remaining_themes:
        if len(roadmap) >= 4:
            break
        roadmap.append({
            "week": len(roadmap) + 1,
            "title": WEEK_THEMES[cat]["title"],
            "tasks": WEEK_THEMES[cat]["tasks"],
        })

    # Ensure exactly 4 weeks
    while len(roadmap) < 4:
        roadmap.append({
            "week": len(roadmap) + 1,
            "title": "General Improvement",
            "tasks": [
                "Review and consolidate learning from previous weeks",
                "Practice mock interviews",
                "Update resume with new skills and projects",
                "Network with alumni and professionals on LinkedIn",
            ],
        })

    logger.info("Generated %d-week roadmap", len(roadmap))
    return roadmap[:4]


# ═══════════════════════════════════════════════════════════════════════════
# Format as Text
# ═══════════════════════════════════════════════════════════════════════════
def format_skill_gap_text(analysis: dict) -> str:
    """Format skill gap analysis as human-readable text.

    Args:
        analysis: Output from ``analyse_skill_gaps()``.

    Returns:
        Formatted string.
    """
    lines = ["📋 Skill Gap Analysis", "=" * 40]

    lines.append("\n✅ Strengths:")
    for s in analysis["strengths"]:
        lines.append(f"   ✔ {s['category']} (score: {s['score']:.2f})")

    lines.append("\n⚠️  Weaknesses:")
    for w in analysis["weaknesses"]:
        lines.append(f"   ✖ {w['category']} (score: {w['score']:.2f})")
        for sug in w["suggestions"]:
            lines.append(f"     → {sug}")

    return "\n".join(lines)


def format_roadmap_text(roadmap: list[dict]) -> str:
    """Format roadmap as human-readable text.

    Args:
        roadmap: Output from ``generate_roadmap()``.

    Returns:
        Formatted string.
    """
    lines = ["🗺️  Personalized 4-Week Roadmap", "=" * 40]

    for week in roadmap:
        lines.append(f"\n📅 Week {week['week']}: {week['title']}")
        for task in week["tasks"]:
            lines.append(f"   • {task}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Example usage
    sample = {
        "CGPA": 0.7, "CodingScore": 0.8, "Aptitude": 0.6,
        "Projects": 0.5, "Internship": 0.0, "Communication": 0.3,
        "Hackathons": 0.2, "GitHub": 1.0, "Certifications": 0.1,
        "LeadershipScore": 0.4, "Attendance": 0.8, "Backlogs": 0.0,
        "AcademicScore": 0.7, "TechnicalScore": 0.7,
        "ExperienceScore": 0.1, "CommunicationScore": 0.3,
        "ProjectScore": 0.4, "CodingIndex": 0.7, "OverallSkillScore": 0.5,
    }
    analysis = analyse_skill_gaps(sample)
    print(format_skill_gap_text(analysis))
    roadmap = generate_roadmap(analysis["weaknesses"], analysis["strengths"])
    print(format_roadmap_text(roadmap))
