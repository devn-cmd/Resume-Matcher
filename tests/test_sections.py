
from src.sections import extract_entities

SAMPLE = """John Doe

EXPERIENCE
Senior ML Engineer — Acme Corp
Jan 2021 - Present
Built ranking models.

Data Scientist, Beta Inc
2018 - 2020

EDUCATION
M.S. Computer Science
Stanford University, 2018

CERTIFICATIONS
AWS Certified Machine Learning – Specialty, 2022
"""


def test_experience_parsed():
    r = extract_entities(SAMPLE)
    assert len(r.experience) == 2
    assert r.experience[0].title == "Senior ML Engineer"
    assert r.experience[0].organization == "Acme Corp"
    assert r.experience[0].end.lower() == "present"


def test_education_parsed():
    r = extract_entities(SAMPLE)
    assert r.education[0].institution.startswith("Stanford")
    assert r.education[0].year == "2018"


def test_certifications_parsed():
    r = extract_entities(SAMPLE)
    assert any("AWS" in c.name for c in r.certifications)