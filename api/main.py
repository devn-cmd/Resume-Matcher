from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import tempfile, os

from src.ingestion import read_document
from src.matching import ResumeMatcher

app = FastAPI(title="Resume Matcher API")
matcher = ResumeMatcher()


class RankRequest(BaseModel):
    job_description: str
    resumes: dict[str, str]


@app.post("/rank")
def rank(req: RankRequest):
    results = matcher.rank(req.job_description, req.resumes)
    return [{
        "resume_id": r.resume_id,
        "final_score": r.final_score,
        "semantic_score": r.semantic_score,
        "skill_score": r.skill_score,
        "suitable": matcher.is_suitable(r),
        "matched_skills": sorted(r.matched_skills),
        "missing_skills": sorted(r.missing_skills),
    } for r in results]


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
    results = matcher.rank(job_description, resumes)
    return [{"resume_id": r.resume_id, "final_score": r.final_score,
             "suitable": matcher.is_suitable(r)} for r in results]

app = FastAPI(title="Resume Matcher API")
matcher = ResumeMatcher()



@app.get("/")
def home():
    return {
        "service": "Resume Matcher API",
        "message": "This API ranks candidate resumes against a job description.",
        "how_to_use": "Open http://127.0.0.1:8000/docs in your browser to test it.",
        "endpoints": {
            "/rank": "Send a job description + resume texts as JSON, get ranked results.",
            "/rank-files": "Upload a job description + resume files (PDF/DOCX/TXT), get ranked results.",
            "/docs": "Interactive page to try the API in your browser."
        }
    }


class RankRequest(BaseModel):
    ...