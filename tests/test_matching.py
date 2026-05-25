from src.matching import ResumeMatcher

def test_relevant_ranks_first():
    m = ResumeMatcher()
    jd = "Looking for a Python machine learning engineer with NLP experience."
    resumes = {
        "good": "Python developer, machine learning, NLP, deep learning, PyTorch.",
        "bad":  "Hotel front desk receptionist with customer service skills.",
    }
    ranked = m.rank(jd, resumes)
    assert ranked[0].resume_id == "good"