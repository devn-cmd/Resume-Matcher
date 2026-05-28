"""Loads the evaluation fixtures into the shape both `evaluate.py` and the
active-learning harness expect:

    jd_texts : {jd_id -> jd_text}
    resumes  : {resume_id -> resume_text}
    labels   : DataFrame[jd_id, resume_id, relevant]

Project layout (from the repo):
    data/raw/resumes/*.(pdf|docx|txt)   — résumé files
    data/eval/label.csv                 — relevance labels
    data/eval/jds.csv                   — job descriptions keyed by jd_id

The loader auto-detects column names and reconciles whether `resume_id` in the
labels uses the full filename or just the stem, then prints how many labelled
pairs actually resolved to a résumé file. If something doesn't line up, the
diagnostics tell you exactly what to fix.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- configurable paths (defaults match the described layout) ----
RESUME_DIR = Path("data/raw/resumes")
EVAL_DIR   = Path("data/eval")

_RESUME_EXTS = {".pdf", ".docx", ".txt"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _find_csv(directory: Path, *needles: str) -> Path:
    """Case-insensitive search for a CSV in `directory` whose name contains any
    of `needles`. Returns the first match."""
    if not directory.exists():
        raise FileNotFoundError(f"Eval directory not found: {directory.resolve()}")
    csvs = list(directory.glob("*.csv")) + list(directory.glob("*.CSV"))
    for needle in needles:
        for p in csvs:
            if needle.lower() in p.name.lower():
                return p
    raise FileNotFoundError(
        f"No CSV matching {needles} in {directory.resolve()}. "
        f"Found: {[p.name for p in csvs]}"
    )


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_labels(path: Path | None = None) -> pd.DataFrame:
    path = path or _find_csv(EVAL_DIR, "label")
    df = pd.read_csv(path)
    jd   = _col(df, "jd_id", "job_id", "jd")
    rid  = _col(df, "resume_id", "candidate_id", "resume", "candidate")
    rel  = _col(df, "relevant", "label", "is_relevant", "relevance")
    missing = [n for n, v in [("jd_id", jd), ("resume_id", rid), ("relevant", rel)] if v is None]
    if missing:
        raise ValueError(
            f"label CSV ({path.name}) is missing column(s) for {missing}. "
            f"Columns present: {list(df.columns)}"
        )
    out = df[[jd, rid, rel]].rename(columns={jd: "jd_id", rid: "resume_id", rel: "relevant"})
    out["jd_id"] = out["jd_id"].astype(str)
    out["resume_id"] = out["resume_id"].astype(str)
    out["relevant"] = out["relevant"].astype(int)
    return out


def load_jds(path: Path | None = None) -> dict[str, str]:
    path = path or _find_csv(EVAL_DIR, "jds", "jd")
    df = pd.read_csv(path)
    jd = _col(df, "jd_id", "job_id", "id", "jd")
    text = _col(df, "jd_text", "text", "description", "job_description", "content", "jd")
    # If the text column collided with the id column, pick the most text-like one.
    if text is None or text == jd:
        non_id = [c for c in df.columns if c != jd]
        text = max(non_id, key=lambda c: df[c].astype(str).str.len().mean()) if non_id else None
    if jd is None or text is None:
        raise ValueError(
            f"Couldn't identify id/text columns in {path.name}. "
            f"Columns present: {list(df.columns)}"
        )
    return {str(k): str(v) for k, v in zip(df[jd], df[text])}


def load_resumes_from_csv(path: Path | None = None) -> dict[str, str]:
    """Read the 60 evaluation résumé texts from a CSV keyed by resume_id.

    This is the benchmark corpus (resume_0000 ...), stored as text rows — not
    the demo files in data/raw/resumes.
    """
    path = path or _find_csv(EVAL_DIR, "resume")
    df = pd.read_csv(path)
    rid = _col(df, "resume_id", "candidate_id", "id", "resume")
    text = _col(df, "resume_text", "text", "content", "cv", "body", "resume")
    # If the text column collided with the id column, pick the most text-like one.
    if text is None or text == rid:
        non_id = [c for c in df.columns if c != rid]
        text = max(non_id, key=lambda c: df[c].astype(str).str.len().mean()) if non_id else None
    if rid is None or text is None:
        raise ValueError(
            f"Couldn't identify id/text columns in {path.name}. "
            f"Columns present: {list(df.columns)}"
        )
    return {str(k): str(v) for k, v in zip(df[rid], df[text])}


def load_resumes_from_files(directory: Path | None = None) -> dict[str, str]:
    """Fallback: read résumé files from a folder (e.g. data/raw/resumes)."""
    directory = directory or RESUME_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Resume directory not found: {directory.resolve()}")
    from src.ingestion import read_document   # lazy: only needed here
    out: dict[str, str] = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in _RESUME_EXTS:
            try:
                out[f.name] = read_document(str(f))
            except Exception as e:               # noqa: BLE001
                print(f"[eval_data] skipped {f.name}: {e}")
    if not out:
        raise FileNotFoundError(f"No PDF/DOCX/TXT résumés found in {directory.resolve()}")
    return out


def load_resumes() -> dict[str, str]:
    """CSV-first: read the benchmark résumé texts from data/eval/resume.csv.
    Falls back to the data/raw/resumes folder if no résumé CSV is present."""
    try:
        path = _find_csv(EVAL_DIR, "resume")
        return load_resumes_from_csv(path)
    except FileNotFoundError:
        print("[eval_data] no résumé CSV in data/eval; falling back to file folder")
        return load_resumes_from_files()


def _reconcile_keys(resumes: dict[str, str], label_ids: set[str]) -> dict[str, str]:
    """Match the résumé dict's keys to the labels' resume_id convention.

    Handles three common cases: labels use the full filename, labels use the
    stem (no extension), or labels use a different extension casing.
    """
    if label_ids & set(resumes):                 # full filename matches
        return resumes
    by_stem = {Path(k).stem: v for k, v in resumes.items()}
    if label_ids & set(by_stem):                 # stem matches
        return by_stem
    # last resort: case-insensitive stem
    by_lower_stem = {Path(k).stem.lower(): v for k, v in resumes.items()}
    if {i.lower() for i in label_ids} & set(by_lower_stem):
        return {k: v for k, v in by_lower_stem.items()}
    return resumes                               # no match — coverage report will flag it


def load_eval_data():
    """Return (jd_texts, resumes, labels) for the harness / evaluate.py."""
    labels = load_labels()
    jd_texts = load_jds()
    resumes = load_resumes()
    resumes = _reconcile_keys(resumes, set(labels["resume_id"]))

    # ---- coverage diagnostics ----
    n_pairs = len(labels)
    have_resume = labels["resume_id"].isin(resumes).sum()
    have_jd = labels["jd_id"].isin(jd_texts).sum()
    print(f"[eval_data] {len(jd_texts)} JDs, {len(resumes)} résumés, {n_pairs} labelled pairs")
    print(f"[eval_data] pairs whose résumé resolved: {have_resume}/{n_pairs}")
    print(f"[eval_data] pairs whose JD resolved:     {have_jd}/{n_pairs}")
    if have_resume == 0:
        sample_lbl = list(labels['resume_id'].head(3))
        sample_file = list(resumes.keys())[:3]
        print(f"[eval_data] WARNING: no résumé IDs matched. "
              f"label resume_id e.g. {sample_lbl}; résumé keys e.g. {sample_file}. "
              f"Adjust how resume_id is written in label.csv (filename vs stem).")
    return jd_texts, resumes, labels


if __name__ == "__main__":
    jd_texts, resumes, labels = load_eval_data()
    print("\nSample JD ids:", list(jd_texts)[:5])
    print("Sample résumé ids:", list(resumes)[:5])
    print(labels.head())