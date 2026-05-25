"""
prepare_eval.py  — Phase 6 data prep

Reads the Kaggle 'Updated Resume Dataset' CSV (Category + Resume columns),
cleans the mojibake/encoding junk, picks a few categories to act as job
descriptions, and writes the three files the evaluation harness needs:

    data/eval/resumes.csv   ->  resume_id, text   (cleaned resume text)
    data/eval/jds.csv       ->  jd_id, text       (one job description per role)
    data/eval/labels.csv    ->  resume_id, jd_id, relevant   (1/0 ground truth)

Run from the project root:
    python data/eval/prepare_eval.py
"""

from pathlib import Path
import re
import pandas as pd

# ---- ftfy cleans the double-encoded "mojibake" (the Ã¢Â€Â¢ junk) -------------
try:
    from ftfy import fix_text
except ImportError:
    # Fallback if ftfy isn't installed: undo the common double-encoding.
    def fix_text(s: str) -> str:
        try:
            return s.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return s

# ---- Settings you can tweak --------------------------------------------------
INPUT_CSV = Path("data/eval/UpdatedResumeDataSet.csv")
OUT_DIR = Path("data/eval")
N_PER_CATEGORY = 12          # resumes sampled per role (12 x 5 roles = 60 resumes)
SEED = 42                    # fixed seed => same sample every run (reproducible)

# Roles we'll turn into job descriptions, with a short JD for each.
# The script uses only the ones that actually exist in your CSV.
JD_LIBRARY = {
    "Data Science": (
        "We are hiring a Data Scientist to build machine learning models, work "
        "with NLP on text data, and analyze large datasets using Python, "
        "scikit-learn, pandas, and SQL. Statistics and data visualization required."
    ),
    "Java Developer": (
        "Looking for a Java Developer experienced in Spring Boot, REST APIs, "
        "object-oriented design, SQL databases, and building scalable backend services."
    ),
    "HR": (
        "Seeking an HR professional skilled in recruitment, employee relations, "
        "onboarding, payroll, performance management, and HR policy administration."
    ),
    "Advocate": (
        "Hiring an Advocate with strong litigation, legal research, drafting, and "
        "contract experience, able to represent clients in civil and criminal matters."
    ),
    "Sales": (
        "We need a Sales professional with experience in business development, "
        "client relationship management, negotiation, lead generation, and meeting targets."
    ),
}


def _slug(name: str) -> str:
    """'Data Science' -> 'jd_data_science'."""
    return "jd_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _detect_columns(df: pd.DataFrame):
    """Find the category column and the resume-text column by name."""
    cat_col = text_col = None
    for c in df.columns:
        lc = c.lower()
        if cat_col is None and "categ" in lc:
            cat_col = c
        if text_col is None and ("resume" in lc or "text" in lc):
            text_col = c
    # Fallback: assume first two columns are category, then resume text.
    if cat_col is None:
        cat_col = df.columns[0]
    if text_col is None:
        text_col = df.columns[1]
    return cat_col, text_col


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Could not find {INPUT_CSV}. Put the downloaded CSV there.")

    df = pd.read_csv(INPUT_CSV)
    cat_col, text_col = _detect_columns(df)
    print(f"Using category column = '{cat_col}', text column = '{text_col}'")

    # Clean the text (fixes the mojibake) and drop empty rows.
    df[text_col] = df[text_col].astype(str).map(fix_text).str.strip()
    df = df[df[text_col].str.len() > 0]

    # Keep only the roles that exist in BOTH our JD library and the file.
    present = [c for c in JD_LIBRARY if c in set(df[cat_col].unique())]
    if len(present) < 2:
        raise SystemExit(
            f"Only found these roles in the CSV: {sorted(df[cat_col].unique())[:10]}...\n"
            "Edit JD_LIBRARY to match the category names in your file."
        )
    print(f"Building evaluation for roles: {present}")

    # Sample N resumes per chosen role (reproducible thanks to the seed).
    # Plain loop + concat keeps the category column intact across pandas versions.
    parts = []
    for role in present:
        g = df[df[cat_col] == role]
        parts.append(g.sample(min(len(g), N_PER_CATEGORY), random_state=SEED))
    sampled = pd.concat(parts).reset_index(drop=True)
    sampled["resume_id"] = [f"resume_{i:04d}" for i in range(len(sampled))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) resumes.csv  (resume_id, text)
    sampled[["resume_id", text_col]].rename(columns={text_col: "text"}).to_csv(
        OUT_DIR / "resumes.csv", index=False
    )

    # 2) jds.csv  (jd_id, text)
    jd_rows = [{"jd_id": _slug(c), "text": JD_LIBRARY[c]} for c in present]
    pd.DataFrame(jd_rows).to_csv(OUT_DIR / "jds.csv", index=False)

    # 3) labels.csv  (resume_id, jd_id, relevant) — every resume vs every JD
    label_rows = []
    for _, row in sampled.iterrows():
        for role in present:
            label_rows.append({
                "resume_id": row["resume_id"],
                "jd_id": _slug(role),
                "relevant": int(row[cat_col] == role),
            })
    pd.DataFrame(label_rows).to_csv(OUT_DIR / "labels.csv", index=False)

    print(f"Done. Wrote {len(sampled)} resumes, {len(present)} JDs, "
          f"{len(label_rows)} label rows to {OUT_DIR}/")


if __name__ == "__main__":
    main()