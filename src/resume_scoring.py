"""
Resume Scoring — Rule-Based ATS Resume Analyser (Placeholder)
================================================================

Analyses resume text using keyword matching to provide:
  - ATS (Applicant Tracking System) Score
  - Missing Skills
  - Formatting Suggestions
  - Important Keywords Found

⚠️ This is a placeholder module — no LLM APIs are used.
   It uses keyword dictionaries for matching.
"""

import logging
import re
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
# Keyword Dictionaries
# ═══════════════════════════════════════════════════════════════════════════
TECHNICAL_SKILLS: set[str] = {
    "python", "java", "c++", "c", "javascript", "sql", "html", "css",
    "react", "node.js", "django", "flask", "spring", "angular",
    "machine learning", "deep learning", "data science", "ai",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux",
    "mongodb", "mysql", "postgresql", "redis", "api", "rest",
    "data structures", "algorithms", "oop", "system design",
    "agile", "scrum", "devops", "ci/cd", "microservices",
}

ACTION_VERBS: set[str] = {
    "developed", "implemented", "designed", "created", "built",
    "managed", "led", "optimised", "improved", "analysed",
    "deployed", "automated", "integrated", "collaborated",
    "achieved", "reduced", "increased", "delivered", "maintained",
    "architected", "streamlined", "resolved", "mentored",
}

RESUME_SECTIONS: set[str] = {
    "education", "experience", "skills", "projects",
    "certifications", "achievements", "summary", "objective",
    "contact", "languages", "interests", "references",
}

FORMATTING_KEYWORDS: set[str] = {
    "email", "phone", "linkedin", "github", "portfolio",
    "gpa", "cgpa", "university", "college", "degree",
    "bachelor", "master", "b.tech", "m.tech", "b.e", "m.e",
}


# ═══════════════════════════════════════════════════════════════════════════
# Scoring Logic
# ═══════════════════════════════════════════════════════════════════════════
def _tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into words and bigrams."""
    text = text.lower()
    words = re.findall(r"[a-z0-9+#./-]+", text)
    # Also create bigrams for multi-word skills
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def score_resume(
    resume_text: str,
    role: Optional[str] = None,
) -> dict:
    """Analyse resume text and compute an ATS score.

    Args:
        resume_text: Raw text content of the resume.
        role: Target job role (for future enhancement).

    Returns:
        Dict with ``ats_score``, ``found_skills``, ``missing_skills``,
        ``found_sections``, ``missing_sections``, ``action_verbs_found``,
        ``formatting_found``, ``suggestions``.
    """
    if not resume_text or not resume_text.strip():
        logger.warning("Empty resume text provided.")
        return {
            "ats_score": 0,
            "found_skills": [],
            "missing_skills": list(TECHNICAL_SKILLS),
            "suggestions": ["Provide resume text for analysis"],
        }

    tokens = set(_tokenize(resume_text))
    text_lower = resume_text.lower()

    # ── Skill matching ──────────────────────────────────────────────────
    found_skills = sorted([s for s in TECHNICAL_SKILLS if s in tokens or s in text_lower])
    missing_skills = sorted([s for s in TECHNICAL_SKILLS if s not in tokens and s not in text_lower])
    skill_score = min(len(found_skills) / max(len(TECHNICAL_SKILLS) * 0.3, 1), 1.0)

    # ── Section detection ───────────────────────────────────────────────
    found_sections = sorted([s for s in RESUME_SECTIONS if s in text_lower])
    missing_sections = sorted([s for s in RESUME_SECTIONS if s not in text_lower])
    section_score = min(len(found_sections) / max(len(RESUME_SECTIONS) * 0.5, 1), 1.0)

    # ── Action verbs ────────────────────────────────────────────────────
    found_verbs = sorted([v for v in ACTION_VERBS if v in tokens])
    verb_score = min(len(found_verbs) / max(len(ACTION_VERBS) * 0.2, 1), 1.0)

    # ── Formatting keywords ─────────────────────────────────────────────
    found_formatting = sorted([f for f in FORMATTING_KEYWORDS if f in text_lower])
    format_score = min(len(found_formatting) / max(len(FORMATTING_KEYWORDS) * 0.4, 1), 1.0)

    # ── Length check ────────────────────────────────────────────────────
    word_count = len(resume_text.split())
    length_score = 1.0 if 200 <= word_count <= 800 else 0.6

    # ── Composite ATS score ─────────────────────────────────────────────
    ats_score = (
        skill_score * 0.35
        + section_score * 0.20
        + verb_score * 0.15
        + format_score * 0.15
        + length_score * 0.15
    )
    ats_score = round(ats_score * 100, 1)

    # ── Suggestions ─────────────────────────────────────────────────────
    suggestions: list[str] = []
    if len(found_skills) < 5:
        suggestions.append("Add more technical skills to your resume")
    if "projects" not in found_sections:
        suggestions.append("Add a Projects section showcasing your work")
    if "experience" not in found_sections:
        suggestions.append("Add an Experience section (internships, jobs, volunteer work)")
    if len(found_verbs) < 3:
        suggestions.append("Use more action verbs (developed, implemented, designed)")
    if "linkedin" not in text_lower:
        suggestions.append("Include your LinkedIn profile URL")
    if "github" not in text_lower:
        suggestions.append("Include your GitHub profile URL")
    if word_count < 200:
        suggestions.append("Resume is too short — add more details")
    if word_count > 800:
        suggestions.append("Resume is too long — keep it concise (1–2 pages)")
    if "summary" not in found_sections and "objective" not in found_sections:
        suggestions.append("Add a Professional Summary or Objective")
    if not suggestions:
        suggestions.append("Resume looks well-structured — keep it updated!")

    result = {
        "ats_score": ats_score,
        "found_skills": found_skills,
        "missing_skills": missing_skills[:10],  # Top 10 missing
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "action_verbs_found": found_verbs,
        "formatting_found": found_formatting,
        "suggestions": suggestions,
        "word_count": word_count,
    }

    logger.info("ATS Score: %.1f%% | Skills found: %d | Sections: %d",
                ats_score, len(found_skills), len(found_sections))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Format as Text
# ═══════════════════════════════════════════════════════════════════════════
def format_resume_text(analysis: dict) -> str:
    """Format resume analysis as human-readable text.

    Args:
        analysis: Output from ``score_resume()``.

    Returns:
        Formatted string.
    """
    lines = [
        "📄 Resume Analysis (ATS Scoring)",
        "=" * 40,
        f"",
        f"📊 ATS Score: {analysis['ats_score']}%",
        f"📝 Word Count: {analysis.get('word_count', 'N/A')}",
        "",
    ]

    if analysis["found_skills"]:
        lines.append(f"✅ Skills Found ({len(analysis['found_skills'])}):")
        lines.append(f"   {', '.join(analysis['found_skills'][:10])}")

    if analysis["missing_skills"]:
        lines.append(f"\n⚠️  Key Missing Skills:")
        lines.append(f"   {', '.join(analysis['missing_skills'][:8])}")

    if analysis["suggestions"]:
        lines.append(f"\n💡 Suggestions:")
        for sug in analysis["suggestions"]:
            lines.append(f"   → {sug}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample_resume = """
    John Doe
    Email: john@example.com | Phone: 9876543210
    LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

    Summary
    Passionate Computer Science student with expertise in Python, Java, and Machine Learning.

    Education
    B.Tech in Computer Science — XYZ University | CGPA: 8.5

    Experience
    Software Engineering Intern — ABC Corp (Summer 2024)
    - Developed RESTful APIs using Flask and deployed on AWS
    - Implemented automated testing reducing bugs by 30%

    Projects
    - Built a machine learning model for sentiment analysis using scikit-learn
    - Designed a full-stack web application with React and Node.js

    Skills
    Python, Java, SQL, Machine Learning, Docker, Git, Linux, React

    Certifications
    - AWS Cloud Practitioner
    - Google Data Analytics Professional Certificate
    """

    result = score_resume(sample_resume)
    print(format_resume_text(result))
