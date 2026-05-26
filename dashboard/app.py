import os
import tempfile

import pandas as pd
import streamlit as st

from src.ingestion import read_document
from src.matching import ResumeMatcher

# default threshold from config, with a safe fallback
try:
    from config import MATCH_THRESHOLD
except Exception:
    MATCH_THRESHOLD = 0.50

st.set_page_config(page_title="Resume Matcher", layout="wide")


# Load the SBERT model ONCE, not on every rerun (big speed fix)
@st.cache_resource
def get_matcher():
    return ResumeMatcher()


matcher = get_matcher()

# --- Header (Step 5) ---
st.title("Resume Screening & Skill Matching")
st.caption("Upload candidate resumes, paste a job description, and rank by fit.")

# --- JD + uploads side by side (Step 5) ---
col1, col2 = st.columns(2)
with col1:
    jd = st.text_area("Job Description", height=220,
                      placeholder="Paste the job description here...")
with col2:
    uploaded = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

# --- Controls in sidebar (Step 3) ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Match threshold", 0.0, 1.0,
                              float(MATCH_THRESHOLD), 0.01)
only_suitable = st.sidebar.checkbox("Show only suitable candidates")

# --- Guarded ranking; compute once, store in session (Step 2) ---
if st.button("Rank candidates"):
    if not uploaded or not jd.strip():
        st.warning("Please upload at least one resume and enter a job description.")
    else:
        with st.spinner("Ranking candidates..."):
            resumes = {}
            for f in uploaded:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                resumes[f.name] = read_document(path)
                os.unlink(path)
            results = matcher.rank(jd, resumes)
        # store plain data so slider/filter reruns stay cheap (no re-embedding)
        st.session_state["ranking"] = [
            {
                "Candidate": r.resume_id,
                "Score": r.final_score,
                "Semantic": r.semantic_score,
                "Skill match": r.skill_score,
            }
            for r in results
        ]

# --- Render from session: reruns cheaply when slider/filter change ---
if "ranking" in st.session_state:
    df = pd.DataFrame(st.session_state["ranking"])
    df["Suitable"] = df["Score"] >= threshold

    if only_suitable:
        df = df[df["Suitable"]]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Semantic": st.column_config.ProgressColumn(
                "Semantic", min_value=0.0, max_value=1.0, format="%.2f"),
            "Skill match": st.column_config.ProgressColumn(
                "Skill match", min_value=0.0, max_value=1.0, format="%.2f"),
            "Suitable": st.column_config.CheckboxColumn("Suitable"),
        },
    )

    # --- Export (Step 4) ---
    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )