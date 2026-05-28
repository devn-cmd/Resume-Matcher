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
from src.language import language_name
from src.feedback import (
    FeedbackRecord, log_feedback, count_feedback, class_balance, load_feedback,
)
from src.active_learning import (
    ActiveLearningRanker, load_or_train_from_disk, FEATURE_NAMES,
)
# default threshold from config, with a safe fallback
try:
    from config import MATCH_THRESHOLD, ENABLE_ACTIVE_LEARNING
except Exception:
    MATCH_THRESHOLD = 0.50
    ENABLE_ACTIVE_LEARNING = True
 
st.set_page_config(page_title="Resume Matcher", layout="wide")
 
 

# Load the SBERT model ONCE, not on every rerun (big speed fix)
@st.cache_resource
def get_matcher():
    m = ResumeMatcher()
    from src.embeddings import embed_one
    embed_one("warmup")          # loads the model now, once
    return m

@st.cache_resource
def get_ranker():
    """Load or train the active-learning ranker on startup."""
    return load_or_train_from_disk() if ENABLE_ACTIVE_LEARNING else ActiveLearningRanker()
 
 
with st.spinner("Loading multilingual model (first time only)..."):
    matcher = get_matcher()
ranker = get_ranker()

# --- Header (Step 5) ---
st.title("Resume Screening & Skill Matching")
st.caption("Upload candidate resumes, paste a job description, and rank by fit. "
           "Resumes and job descriptions may be in any language.")

@st.cache_resource
def get_extractor():
    db = Path(__file__).resolve().parent.parent / "data" / "skills_db.json"
    return SkillExtractor(str(db))

extractor = get_extractor()
# --- JD + uploads side by side  ---
col1, col2 = st.columns(2)
with col1:
    jd = st.text_area("Job Description", height=220,
                      placeholder="Paste the job description here...")
    jd_id = st.text_input("JD identifier (used to log feedback)",
                          value="jd_session", help="Free-form ID for this JD; used to group feedback.")
with col2:
    uploaded = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

# --- Sidebar: settings + active-learning status panel ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Match threshold", 0.0, 1.0,
                              float(MATCH_THRESHOLD), 0.01)
only_suitable = st.sidebar.checkbox("Show only suitable candidates")
 
st.sidebar.markdown("---")
st.sidebar.header("Active learning")
pos, neg = class_balance()
total_fb = pos + neg
st.sidebar.metric("Feedback collected", total_fb, help=f"{pos} shortlists / {neg} rejects")
if ranker.is_trained():
    st.sidebar.success(f"Adaptive re-ranker active (trained on {ranker.n_train_samples} samples)")
else:
    st.sidebar.info("Adaptive re-ranker not active yet — using static blend.")
 
if st.sidebar.button("Retrain re-ranker on accumulated feedback"):
    df = load_feedback()
    report = ranker.fit(df)
    if report.trained:
        ranker.save()
        st.sidebar.success(f"Retrained on {report.n_samples} samples "
                           f"({report.n_positive}+ / {report.n_negative}-)")
    else:
        st.sidebar.warning(f"Not retrained: {report.reason}")
    st.cache_resource.clear()  # force the ranker to reload on next interaction
 
if ranker.is_trained():
    with st.sidebar.expander("Learned coefficients (interpretability)"):
        coefs = ranker.coefficients()
        for name in FEATURE_NAMES:
            st.write(f"`{name}`: **{coefs.get(name, 0.0):+.3f}**")
        st.caption("Positive ⇒ feature pushes toward 'suitable'. "
                   "Linear model — these weights *are* the explanation.")
# --- Guarded ranking ---
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
            # Use the active-learning aware ranking; falls back to static when untrained.
            results = matcher.rank_with_feedback(jd, resumes, ranker=ranker)
 
            ranking = []
            jd_language = results[0].jd_language if results else "en"
            for r in results:
                info = explain(resumes[r.resume_id], set(r.matched_skills),
                               set(r.missing_skills), extractor)
                ranking.append({
                    "Candidate": r.resume_id,
                    "Score": r.final_score,
                    "Semantic": r.semantic_score,
                    "Skill match": r.skill_score,
                    "Adaptive": r.adaptive_score,             # may be None
                    "Preference": r.recruiter_preference_score,
                    "Uncertain": r.is_uncertain,
                    "language": r.language,
                    "cross_lingual": r.cross_lingual,
                    "matched": list(r.matched_skills),
                    "missing": list(r.missing_skills),
                    "num_matched": len(r.matched_skills),
                    "num_missing": len(r.missing_skills),
                    "evidence": info["evidence"],
                    "experience": [e.raw for e in info["structured"].experience][:3],
                    "education": [e.raw for e in info["structured"].education][:3],
                    "recommendations": info.get("recommendations", []),
                })
        st.session_state["ranking"] = ranking
        st.session_state["jd_language"] = jd_language
        st.session_state["jd_id"] = jd_id
 
 
def _log_action(r: dict, action: str):
    """Persist a recruiter action as a feedback record."""
    rec = FeedbackRecord(
        jd_id=st.session_state.get("jd_id", "jd_session"),
        resume_id=r["Candidate"],
        semantic_score=float(r["Semantic"]),
        skill_score=float(r["Skill match"]),
        num_matched=int(r["num_matched"]),
        num_missing=int(r["num_missing"]),
        cross_lingual=int(bool(r["cross_lingual"])),
        matched_skills=[str(s) for s in r["matched"]],
        action=action,
    )
    log_feedback(rec)
 
 
if "ranking" in st.session_state:
    ranking = st.session_state["ranking"]
    jd_language = st.session_state.get("jd_language", "en")
    st.info(f"Job description language detected: **{language_name(jd_language)}**")
 
    for r in ranking:
        # Use adaptive_score for suitability when available; static otherwise.
        score_for_decision = r["Adaptive"] if r["Adaptive"] is not None else r["Score"]
        r["suitable"] = score_for_decision >= threshold
    shown = [r for r in ranking if r["suitable"] or not only_suitable]
 
    # ---- Review queue (active-learning surface) ----
    queue = [r for r in ranking if r["Uncertain"]]
    if queue:
        st.subheader("🔍 Recruiter review queue (uncertain — your label helps most here)")
        st.caption(
            "These candidates sit closest to the decision boundary. Labelling them "
            "improves the re-ranker faster than labelling obvious cases."
        )
        for r in queue:
            cols = st.columns([3, 1, 1, 1])
            tag = " 🌐" if r["cross_lingual"] else ""
            cols[0].markdown(f"**{r['Candidate']}**{tag} — "
                             f"static {r['Score']:.2f}"
                             + (f" · adaptive {r['Adaptive']:.2f}" if r['Adaptive'] is not None else ""))
            if cols[1].button("✅ Shortlist", key=f"sl_{r['Candidate']}"):
                _log_action(r, "shortlist"); st.toast("Logged: shortlist"); st.rerun()
            if cols[2].button("❌ Reject", key=f"rj_{r['Candidate']}"):
                _log_action(r, "reject");    st.toast("Logged: reject");    st.rerun()
            cols[3].caption("uncertain")
 
    # ---- Ranked table ----
    df = pd.DataFrame([{
        "Candidate": r["Candidate"],
        "Language": language_name(r["language"]) + (" \U0001F310" if r["cross_lingual"] else ""),
        "Score": r["Score"] * 100,
        "Adaptive": (r["Adaptive"] * 100) if r["Adaptive"] is not None else None,
        "Semantic": r["Semantic"] * 100,
        "Skill match": r["Skill match"] * 100,
        "Suitable": "\u2705" if r["suitable"] else "\u274C",
    } for r in shown])
    st.dataframe(
        df, width='stretch', hide_index=True,
        column_config={
            "Score":      st.column_config.ProgressColumn("Static", min_value=0, max_value=100, format="%.0f%%"),
            "Adaptive":   st.column_config.ProgressColumn("Adaptive", min_value=0, max_value=100, format="%.0f%%"),
            "Semantic":   st.column_config.ProgressColumn("Semantic", min_value=0, max_value=100, format="%.0f%%"),
            "Skill match":st.column_config.ProgressColumn("Skill match", min_value=0, max_value=100, format="%.0f%%"),
        },
    )
    st.caption("\U0001F310 = candidate's language differs from the job description. "
               "Adaptive column is blank until the re-ranker is trained.")
    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ranked_candidates.csv", mime="text/csv",
    )
 
    # ---- Skill breakdown with per-candidate feedback buttons ----
    st.subheader("Skill breakdown")
    for r in shown:
        flag = "\u2705" if r["suitable"] else "\u274C"
        lang_tag = language_name(r["language"]) + (" \U0001F310" if r["cross_lingual"] else "")
        with st.expander(f"{flag}  {r['Candidate']}  ·  {lang_tag}"):
            # Feedback buttons live with the candidate for context.
            bcols = st.columns([1, 1, 4])
            if bcols[0].button("✅ Shortlist", key=f"slx_{r['Candidate']}"):
                _log_action(r, "shortlist"); st.toast("Logged: shortlist"); st.rerun()
            if bcols[1].button("❌ Reject", key=f"rjx_{r['Candidate']}"):
                _log_action(r, "reject");    st.toast("Logged: reject");    st.rerun()
            bcols[2].caption("Your judgement is stored as a feature vector + label "
                             "(no raw resume text retained).")
 
            st.markdown("**Matched skills (with evidence):**")
            if r["matched"]:
                for skill in r["matched"]:
                    sent = r["evidence"].get(skill)
                    st.markdown(f"- **{skill}**" + (f" — _{sent}_" if sent else ""))
            else:
                st.markdown("_none_")
 
            st.markdown(f"**Missing skills:** {', '.join(r['missing']) if r['missing'] else '—'}")
 
            # --- Upskilling roadmaps ---
            if r.get("recommendations"):
                st.markdown("---")
                st.markdown("🚀 **Suggested Upskilling Roadmap:**")
                for rec in r["recommendations"]:
                    st.markdown(f"**{rec['skill']}** *(Market Demand: {rec['demand_score']})*")
                    for step in rec['roadmap']:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **Step {step['step']}:** "
                                    f"{step['action']} — [{step['resource']}]")
                st.markdown("---")
 
            if r["experience"]:
                st.markdown("**Experience:**")
                for e in r["experience"]:
                    st.markdown(f"- {e}")
            if r["education"]:
                st.markdown("**Education:**")
                for e in r["education"]:
                    st.markdown(f"- {e}")