import streamlit as st
import pandas as pd, tempfile, os
from src.ingestion import read_document
from src.matching import ResumeMatcher

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Screening & Skill Matching")
jd = st.text_area("Job Description", height=180)
uploaded = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"],
                            accept_multiple_files=True)

matcher = ResumeMatcher()

if st.button("Rank candidates") and jd and uploaded:
    resumes = {}
    for f in uploaded:
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read()); path = tmp.name
        resumes[f.name] = read_document(path); os.unlink(path)

    results = matcher.rank(jd, resumes)
    df = pd.DataFrame([{
        "Candidate": r.resume_id, "Score": r.final_score,
        "Semantic": r.semantic_score, "Skill match": r.skill_score,
        "Suitable": "✅" if matcher.is_suitable(r) else "—",
    } for r in results])
    st.dataframe(df, use_container_width=True)

