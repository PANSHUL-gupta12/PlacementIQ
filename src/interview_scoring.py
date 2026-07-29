"""
Interview Scoring — Mock Interview Evaluation (Placeholder)
==============================================================

Evaluates mock interview performance based on weighted scores:
  - Question scores (DSA / technical questions)
  - Communication score
  - Confidence score
  - Technical knowledge score

Outputs:
  - Interview Readiness level
  - Weak areas
  - Personalised suggestions

⚠️ This is a placeholder module — uses weighted scoring, not AI.
"""

import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
WEIGHTS: dict[str, float] = {
    "technical": 0.35,
    "communication": 0.25,
    "confidence": 0.20,
    "question_accuracy": 0.20,
}

READINESS_LEVELS: list[tuple[float, str]] = [
    (85, "Excellent — Interview Ready"),
    (70, "Good — Minor Improvements Needed"),
    (55, "Average — Significant Practice Required"),
    (40, "Below Average — Intensive Preparation Needed"),
    (0, "Needs Major Improvement"),
]

AREA_SUGGESTIONS: dict[str, list[str]] = {
    "technical": [
        "Revise core CS fundamentals: OOP, DBMS, OS, Networking",
        "Practice system design questions",
        "Study common interview topics: sorting, trees, graphs, DP",
        "Review your project implementations — be ready to explain them",
    ],
    "communication": [
        "Practice explaining technical concepts in simple terms",
        "Record yourself answering questions and review",
        "Use the STAR method for behavioural questions",
        "Slow down while speaking — clarity over speed",
    ],
    "confidence": [
        "Do more mock interviews (Pramp, Interviewing.io)",
        "Prepare a strong self-introduction (60 seconds)",
        "Research the company before the interview",
        "Practice maintaining eye contact and steady voice",
    ],
    "question_accuracy": [
        "Solve 5 coding problems daily on LeetCode/HackerRank",
        "Time yourself — most interviews give 20-30 minutes per problem",
        "Practice thinking aloud while solving problems",
        "Review optimal solutions after each problem",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Scoring Logic
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_interview(
    technical_score: float,
    communication_score: float,
    confidence_score: float,
    question_accuracy: float,
) -> dict:
    """Evaluate mock interview performance.

    All scores should be on a 0–100 scale.

    Args:
        technical_score: Technical knowledge score (0–100).
        communication_score: Communication clarity score (0–100).
        confidence_score: Confidence and body language score (0–100).
        question_accuracy: Question-answering accuracy score (0–100).

    Returns:
        Dict with ``overall_score``, ``readiness_level``, ``weak_areas``,
        ``strong_areas``, ``suggestions``, ``breakdown``.
    """
    # Validate inputs
    scores = {
        "technical": max(0, min(100, technical_score)),
        "communication": max(0, min(100, communication_score)),
        "confidence": max(0, min(100, confidence_score)),
        "question_accuracy": max(0, min(100, question_accuracy)),
    }

    # Weighted overall score
    overall = sum(scores[area] * WEIGHTS[area] for area in scores)
    overall = round(overall, 1)

    # Readiness level
    readiness = "Needs Major Improvement"
    for threshold, level in READINESS_LEVELS:
        if overall >= threshold:
            readiness = level
            break

    # Identify weak and strong areas
    area_threshold = 60
    weak_areas = [area for area, score in scores.items() if score < area_threshold]
    strong_areas = [area for area, score in scores.items() if score >= area_threshold]

    # Compile suggestions for weak areas
    suggestions: list[str] = []
    for area in weak_areas:
        if area in AREA_SUGGESTIONS:
            suggestions.extend(AREA_SUGGESTIONS[area][:2])  # Top 2 per area

    if not suggestions:
        suggestions.append("Great performance! Keep practising to maintain your edge.")

    result = {
        "overall_score": overall,
        "readiness_level": readiness,
        "weak_areas": weak_areas,
        "strong_areas": strong_areas,
        "suggestions": suggestions,
        "breakdown": scores,
    }

    logger.info("Interview Score: %.1f — %s", overall, readiness)
    logger.info("Weak areas: %s | Strong areas: %s", weak_areas, strong_areas)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Format as Text
# ═══════════════════════════════════════════════════════════════════════════
def format_interview_text(result: dict) -> str:
    """Format interview evaluation as human-readable text.

    Args:
        result: Output from ``evaluate_interview()``.

    Returns:
        Formatted string.
    """
    lines = [
        "🎤 Mock Interview Evaluation",
        "=" * 40,
        "",
        f"📊 Overall Score: {result['overall_score']}/100",
        f"🎯 Readiness: {result['readiness_level']}",
        "",
        "📋 Breakdown:",
    ]

    for area, score in result["breakdown"].items():
        emoji = "✅" if score >= 60 else "⚠️ "
        lines.append(f"   {emoji} {area.replace('_', ' ').title()}: {score}/100")

    if result["strong_areas"]:
        lines.append(f"\n✅ Strong Areas: {', '.join(a.replace('_', ' ').title() for a in result['strong_areas'])}")

    if result["weak_areas"]:
        lines.append(f"⚠️  Weak Areas: {', '.join(a.replace('_', ' ').title() for a in result['weak_areas'])}")

    if result["suggestions"]:
        lines.append("\n💡 Suggestions:")
        for sug in result["suggestions"]:
            lines.append(f"   → {sug}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = evaluate_interview(
        technical_score=72,
        communication_score=55,
        confidence_score=60,
        question_accuracy=68,
    )
    print(format_interview_text(result))
