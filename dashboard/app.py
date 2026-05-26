import os
import tempfile

import pandas as pd
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import read_document
from src.matching import ResumeMatcher
from src.skills import SkillExtractor
from src.explain import explain
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

@st.cache_resource
def get_extractor():
    db = Path(__file__).resolve().parent.parent / "data" / "skills_db.json"
    return SkillExtractor(str(db))

extractor = get_extractor()
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
            ranking = []
            for r in results:
                info = explain(resumes[r.resume_id], set(r.matched_skills), extractor)
                ranking.append({
                    "Candidate": r.resume_id,
                    "Score": r.final_score,
                    "Semantic": r.semantic_score,
                    "Skill match": r.skill_score,
                    "matched": list(r.matched_skills),
                    "missing": list(r.missing_skills),
                    "evidence": info["evidence"],
                    "experience": [e.raw for e in info["structured"].experience][:3],
                    "education": [e.raw for e in info["structured"].education][:3],
                })
        st.session_state["ranking"] = ranking

if "ranking" in st.session_state:
    ranking = st.session_state["ranking"]
    for r in ranking:
        r["suitable"] = r["Score"] >= threshold
    shown = [r for r in ranking if r["suitable"] or not only_suitable]

    df = pd.DataFrame([{
        "Candidate": r["Candidate"],
        "Score": r["Score"],
        "Semantic": r["Semantic"],
        "Skill match": r["Skill match"],
        "Suitable": "\u2705" if r["suitable"] else "\u274C",
    } for r in shown])

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Semantic": st.column_config.ProgressColumn("Semantic", min_value=0.0, max_value=1.0, format="%.2f"),
            "Skill match": st.column_config.ProgressColumn("Skill match", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_candidates.csv", mime="text/csv",
    )

    st.subheader("Skill breakdown")
    for r in shown:
        flag = "\u2705" if r["suitable"] else "\u274C"
        with st.expander(f"{flag}  {r['Candidate']}"):
            st.markdown("**Matched skills (with evidence):**")
            if r["matched"]:
                for skill in r["matched"]:
                    sent = r["evidence"].get(skill)
                    st.markdown(f"- **{skill}**" + (f" — _{sent}_" if sent else ""))
            else:
                st.markdown("_none_")
            st.markdown(f"**Missing skills:** {', '.join(r['missing']) if r['missing'] else '—'}")
            if r["experience"]:
                st.markdown("**Experience:**")
                for e in r["experience"]:
                    st.markdown(f"- {e}")
            if r["education"]:
                st.markdown("**Education:**")
                for e in r["education"]:
                    st.markdown(f"- {e}")