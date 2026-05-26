import re
from src.sections import extract_entities
from src.recommendations import SkillRecommender

# Initialize the recommender at the module level
recommender = SkillRecommender()

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def skill_evidence(resume_text: str, skills: set, extractor) -> dict:
    """Map each skill to the first resume sentence that mentions it."""
    evidence, remaining = {}, set(skills)
    for sent in _sentences(resume_text):
        if not remaining:
            break
        for skill in extractor.extract(sent) & remaining:
            evidence[skill] = sent
        remaining -= set(evidence)
    return evidence


# CHANGED: Added missing_skills to the parameters
def explain(resume_text: str, matched_skills: set, missing_skills: set, extractor) -> dict:
    """Bundle matched-skill evidence, roadmap recommendations, and structured resume sections."""
    return {
        "evidence": skill_evidence(resume_text, matched_skills, extractor),
        "structured": extract_entities(resume_text),
        # NEW: Generate and include the learning paths here!
        "recommendations": recommender.generate_roadmap(missing_skills)
    }