from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile, os


from src.ingestion import read_document
from src.matching import ResumeMatcher
from src.language import language_name
from src.feedback import (
    FeedbackRecord, log_feedback, count_feedback, class_balance, load_feedback,
)
from src.active_learning import ActiveLearningRanker, load_or_train_from_disk
 
app = FastAPI(title="Resume Matcher API")
matcher = ResumeMatcher()

# Process-wide singletons. The ranker is loaded once on startup, then mutated
# in-place by /retrain so subsequent /rank calls see the updated model.
ranker: ActiveLearningRanker = load_or_train_from_disk()
 

 # Schemas

class RankRequest(BaseModel):
    job_description: str
    resumes: dict[str, str]

class FeedbackRequest(BaseModel):
    jd_id: str
    resume_id: str
    semantic_score: float = Field(ge=0.0, le=1.0)
    skill_score: float = Field(ge=0.0, le=1.0)
    num_matched: int = Field(ge=0)
    num_missing: int = Field(ge=0)
    cross_lingual: bool = False
    matched_skills: List[str] = []
action: str = Field(pattern="^(shortlist|reject)$")
# Root


@app.get("/")
def home():
    return {
        "service": "Resume Matcher API",
        "message": "This API ranks candidate resumes against a job description.",
        "how_to_use": "Open http://127.0.0.1:8000/docs in your browser to test it.",
        "endpoints": {
            "/rank":                   "JSON-in: rank resume strings against a JD.",
            "/rank-files":             "Multipart upload: rank resume files (PDF/DOCX/TXT).",
            "/feedback":               "Log one recruiter judgement (shortlist/reject).",
            "/retrain":                "Retrain the adaptive re-ranker on accumulated feedback.",
            "/active-learning/status": "Inspect feedback counts, model state, learned coefficients.",
            "/docs":                   "Interactive Swagger UI.",
        },
    }
 
 # Ranking — now returns adaptive_score / is_uncertain alongside the existing fields

def _serialise(r):
    return {
        "resume_id":      r.resume_id,
        "final_score":    r.final_score,
        "semantic_score": r.semantic_score,
        "skill_score":    r.skill_score,
        "suitable":       matcher.is_suitable(r),
        "matched_skills": sorted(r.matched_skills),
        "missing_skills": sorted(r.missing_skills),
        # multilingual fields
        "language":       r.language,
        "language_name":  language_name(r.language),
        "jd_language":    r.jd_language,
        "cross_lingual":  r.cross_lingual,
        # active-learning fields (None / False when re-ranker is not yet trained)
        "adaptive_score":             r.adaptive_score,
        "recruiter_preference_score": r.recruiter_preference_score,
        "is_uncertain":               r.is_uncertain,
    }
 
 
@app.post("/rank")
def rank(req: RankRequest):
    results = matcher.rank_with_feedback(req.job_description, req.resumes, ranker=ranker)
    return [_serialise(r) for r in results]
  

@app.post("/rank-files")
async def rank_files(job_description: str = Form(...), files: list[UploadFile] = File(...)):
    resumes = {}
    for f in files:
        suffix = os.path.splitext(f.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        resumes[f.filename] = read_document(tmp_path)
        os.unlink(tmp_path)
    results = matcher.rank_with_feedback(job_description, resumes, ranker=ranker)
    return [_serialise(r) for r in results]
 
 # Active-learning endpoints


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Log one recruiter judgement. Stores features + label only (no raw text)."""
    rec = FeedbackRecord(
        jd_id=req.jd_id,
        resume_id=req.resume_id,
        semantic_score=req.semantic_score,
        skill_score=req.skill_score,
        num_matched=req.num_matched,
        num_missing=req.num_missing,
        cross_lingual=int(req.cross_lingual),
        matched_skills=req.matched_skills,
        action=req.action,
    )
    log_feedback(rec)
    pos, neg = class_balance()
    return {
        "ok": True,
        "total_feedback": pos + neg,
        "positives": pos,
        "negatives": neg,
        "message": "Stored. Call /retrain when you'd like the model to pick this up.",
    }
 
 
@app.post("/retrain")
def retrain():
    """Retrain the adaptive re-ranker on accumulated feedback."""
    df = load_feedback()
    if df.empty:
        raise HTTPException(status_code=400, detail="No feedback has been logged yet.")
    report = ranker.fit(df)
    if not report.trained:
        return {"trained": False, "reason": report.reason,
                "samples": report.n_samples,
                "positives": report.n_positive, "negatives": report.n_negative}
    ranker.save()
    return {
        "trained": True,
        "samples": report.n_samples,
        "positives": report.n_positive,
        "negatives": report.n_negative,
        "coefficients": report.coefficients,
    }
 
 
@app.get("/active-learning/status")
def status():
    pos, neg = class_balance()
    return {
        "feedback_total": pos + neg,
        "feedback_positives": pos,
        "feedback_negatives": neg,
        "model_trained": ranker.is_trained(),
        "model_train_samples": ranker.n_train_samples,
        "coefficients": ranker.coefficients(),
        "top_preferred_skills": sorted(
            ranker.skill_weights.items(), key=lambda kv: -kv[1]
        )[:10],
    }