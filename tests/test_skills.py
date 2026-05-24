# tests/test_skills.py
from src.skills import SkillExtractor, skill_overlap

def test_synonym_resolution():
    ext = SkillExtractor("data/skills_db.json")
    assert "machine learning" in ext.extract("Experienced in ML and Python")

def test_overlap_score():
    score, matched, missing = skill_overlap({"python", "sql"}, {"python", "aws"})
    assert matched == {"python"} and missing == {"aws"}
    assert 0 < score < 1