## Resume Matcher 🚀

## Steamlit DashBoard    :https://resume-matcher-wsgvqrcy3vuinhxgzkvddo.streamlit.app
## Unlisted Youtube Vedio:
## Google Colab          :
## GitHub Repo           :

# For testing resumes and job description are in my data folder


> **An AI-powered Resume Screening & Candidate Ranking System** that combines semantic NLP matching, explicit skill extraction, multilingual embeddings, explainable ranking, recruiter-feedback-driven active learning, and market-driven upskilling recommendations.

The project was developed incrementally using a real-world engineering workflow with modular architecture, evaluation pipelines, feature branches, and reproducible experimentation.

---

## 📌 Project Overview

Resume Matcher automatically evaluates candidate resumes against a job description (JD) and produces:

- **Ranked candidate shortlists** — ordered by relevance
- **Suitability classification** — binary yes/no based on a tunable threshold
- **Explainable scoring breakdowns** — matched skills, missing skills, and evidence sentences
- **Skill gap analysis** — structured gap reports per candidate
- **Personalized upskilling recommendations** — market-driven learning roadmaps with demand scores
- **Multilingual cross-language matching** — English ↔ German/French/Spanish without translation
- **Recruiter feedback-driven active learning** — the model improves as recruiters label uncertain candidates

The system combines:

- **Semantic similarity** using SBERT embeddings
- **Explicit skill overlap** scoring with synonym support
- **Structured resume parsing** (experience, education, certifications)
- **Explainability pipelines** with evidence extraction
- **Cross-lingual embeddings** for multilingual matching
- **Interactive recruiter dashboard** (Streamlit)
- **FastAPI service endpoints** for production integration
- **Market-driven recommendation engine** for candidate skill development

---

## ✨ Features

### Core NLP Features

| Feature | Description |
|---------|-------------|
| Multi-format resume ingestion | PDF, DOCX, TXT support |
| Text preprocessing & normalization | PII stripping, whitespace normalization |
| Skill extraction with synonym support | Canonical skills + aliases via spaCy PhraseMatcher |
| SBERT semantic similarity ranking | Dense vector comparison with cosine similarity |
| Explicit skill overlap scoring | Coverage of JD skills found in resume |
| Blended weighted ranking | `0.7 × Semantic + 0.3 × Skill` |
| Threshold-based suitability classification | Tunable cutoff for shortlist/reject |

### Explainability Features

| Feature | Description |
|---------|-------------|
| Matched skills | Skills present in both resume and JD |
| Missing skills | JD-required skills absent from resume |
| Evidence sentence extraction | First resume sentence mentioning each matched skill |
| Structured experience parsing | Title, organization, date range extraction |
| Structured education parsing | Degree, institution, year extraction |
| Certification extraction | Name, issuer, year extraction |
| Transparent score breakdowns | Semantic, skill, and final scores per candidate |

### Upskilling Recommendations 

| Feature | Description |
|---------|-------------|
| Market demand scoring | Static market trends data with demand scores per skill |
| Curated learning roadmaps | Multi-step, personalized learning paths for missing skills |
| Smart generic fallback | Auto-generated roadmap for skills without explicit data |
| Priority ranking | Skills sorted by demand score (highest value first) |
| Top-N recommendations | Configurable number of recommendations per candidate |

### Multilingual Support 🌍

| Feature | Description |
|---------|-------------|
| Cross-lingual resume matching | Match resumes in any supported language to English JDs |
| Language detection | Offline `langdetect` for resumes and JDs (privacy-safe) |
| Shared multilingual embedding space | `paraphrase-multilingual-MiniLM-L12-v2` |
| Language metadata | Surfaced in API responses and dashboard UI |
| Supported pairs | English ↔ German / French / Spanish |

### Active Learning Features 🧠

| Feature | Description |
|---------|-------------|
| Recruiter feedback logging | Append-only JSONL store (features + labels, no raw text) |
| Uncertainty sampling | Surface candidates closest to the decision boundary |
| Logistic regression reranker | Lightweight, interpretable classification layer |
| Preference-aware ranking | Laplace-smoothed skill-level preferences learned from feedback |
| Feature-based explainable reranking | Inspectable coefficients for every feature |
| Learning curve evaluation harness | Simulated-oracle benchmark with held-out test split |

### Dashboard Features

| Feature | Description |
|---------|-------------|
| Streamlit recruiter UI | Drag-and-drop resume upload, JD paste |
| Ranked candidate table | Sortable, filterable results with progress bars |
| Threshold adjustment slider | Real-time suitability filtering |
| Suitable-only filtering | Toggle to show only shortlisted candidates |
| CSV export | Download results for Excel compatibility |
| Cached model loading | `@st.cache_resource` for sub-second reruns |
| Interactive explanation views | Per-candidate skill breakdowns with evidence |
| Review queue | Uncertain candidates flagged for recruiter labeling |
| Feedback buttons | One-click shortlist/reject per candidate |
| Upskilling roadmaps | Personalized skill development paths per candidate |

---

## 🏗️ Project Architecture

```
resume-matcher/
├── api/
│   ├── main.py                 # FastAPI service (rank, feedback, retrain, status)
│   └── __init__.py
│
├── dashboard/
│   └── app.py                  # Streamlit interactive recruiter UI
│
├── data/
│   ├── skills_db.json          # Canonical skills & multilingual aliases
│   ├── market_trends.json      # Market demand scores & curated learning roadmaps
│   ├── recruiter_feedback.jsonl# Append-only recruiter feedback log
│   ├── eval/                   # Evaluation fixtures (resumes.csv, jds.csv, labels.csv)
│   └── raw/                    # Demo resumes (PDF/DOCX/TXT)
│
├── evaluation/
│   ├── evaluate.py             # Core metrics: P/R/F1, MRR, NDCG
│   ├── run_eval.py             # Evaluation runner entrypoint
│   ├── robustness.py           # Noise & damage injection tests
│   ├── sweep_threshold.py      # F1-optimal threshold search
│   ├── active_learning_eval.py # Simulated-oracle learning-curve harness
│   ├── eval_data.py            # Loader for benchmark corpus (60 resumes, 300 labels)
│   ├── multilingual_eval_data.py# Self-contained multilingual test set
│   └── active_learning_curve.csv# Pre-generated learning curve results
│
├── src/
│   ├── ingestion.py            # PDF/DOCX/TXT text extraction
│   ├── preprocessing.py        # PII stripping, token normalization
│   ├── language.py             # Offline language detection (langdetect)
│   ├── skills.py               # Skill extraction & overlap scoring
│   ├── sections.py             # Structured entity extraction (experience/education/cert)
│   ├── embeddings.py           # SBERT embedding generation (cached singleton)
│   ├── matching.py             # Ranking, scoring, and active-learning integration
│   ├── explain.py              # Explainability pipeline (evidence + structured sections)
│   ├── recommendations.py      # Upskilling recommendation engine [NEW]
│   ├── active_learning.py      # Uncertainty sampling + LR reranker
│   └── feedback.py             # Privacy-safe feedback store (JSONL)
│
├── tests/
│   ├── test_ingestion.py       # Document reader sanity checks
│   ├── test_matching.py        # Ranking logic tests
│   ├── test_sections.py        # Section parser tests
│   └── test_skills.py          # Skill extractor tests
│
├── config.py                   # Central configuration (models, thresholds, AL settings)
├── requirements.txt            # Clean runtime dependency list
├── pytest.ini                 # Test configuration
├── .gitignore                 # Excludes trained models, feedback logs, etc.
├── resume_matcher_demo.ipynb  # End-to-end Colab demo notebook
└── README.md                   # This file
```

---

## 📂 Folder & Module Explanation

### `src/` — Core NLP/ML Pipeline

| File | Responsibility |
|------|----------------|
| `ingestion.py` | Reads PDF/DOCX/TXT resumes using `pdfplumber` and `python-docx` |
| `preprocessing.py` | Cleans text (strips emails, phones, URLs), normalizes whitespace, lemmatizes tokens via spaCy |
| `language.py` | Detects document language using offline `langdetect` (deterministic seed for reproducibility) |
| `skills.py` | Skill extraction via spaCy PhraseMatcher with canonical skill database + synonyms; computes skill overlap |
| `sections.py` | Splits resumes into canonical sections (experience, education, certifications, skills, projects, summary) and parses structured entries with regex heuristics |
| `embeddings.py` | Lazy-loaded SBERT model singleton (`paraphrase-multilingual-MiniLM-L12-v2`); L2-normalized embeddings |
| `matching.py` | `ResumeMatcher` class: ranks candidates using blended semantic + skill scores; integrates active-learning reranker via `rank_with_feedback()` |
| `explain.py` | Bundles matched-skill evidence sentences, structured resume sections, and upskilling recommendations |
| `recommendations.py` | `SkillRecommender` class: maps missing skills to market-demand scores and curated learning roadmaps [NEW] |
| `active_learning.py` | `ActiveLearningRanker` (L2-regularized logistic regression), uncertainty sampling, review queue selection, feature engineering |
| `feedback.py` | Append-only JSONL feedback store; `FeedbackRecord` dataclass with feature vectors + labels (no raw resume text) |

### `evaluation/` — Benchmarking & Evaluation

| File | Purpose |
|------|---------|
| `evaluate.py` | Core metrics: Precision, Recall, F1, MRR, NDCG, average latency per resume |
| `run_eval.py` | Entrypoint to run the full evaluation suite |
| `sweep_threshold.py` | Grid search over match thresholds to find F1-optimal cutoff |
| `robustness.py` | Inject noise (typos, truncation, section removal) and measure rank drift |
| `active_learning_eval.py` | Simulated-oracle harness: uncertainty-sampled feedback → retrain → evaluate on held-out test split |
| `eval_data.py` | Loads the 60-resume benchmark corpus with auto-reconciliation of filename vs. stem IDs |
| `multilingual_eval_data.py` | Self-contained multilingual test set (EN/DE/FR/ES resumes × 3 roles) |

### `dashboard/` — Interactive UI

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app: upload resumes, paste JD, rank, filter, export CSV, provide feedback, trigger retraining, view review queue and upskilling roadmaps |

### `api/` — Production Service

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app: `/rank`, `/rank-files`, `/feedback`, `/retrain`, `/active-learning/status`; process-wide singleton ranker loaded on startup |

### `data/` — Static Assets

| File | Purpose |
|------|---------|
| `skills_db.json` | Canonical skills → aliases (e.g., `"machine learning": ["ml", "machine learning"]`) |
| `market_trends.json` | Skill demand scores + step-by-step learning roadmaps |
| `recruiter_feedback.jsonl` | Seeded feedback log for reproducible active-learning experiments |
| `eval/` | Benchmark corpus: `resumes.csv`, `jds.csv`, `labels.csv`, `prepare_eval.py` |
| `raw/` | Demo files for quick manual testing |

---

## 🔄 End-to-End Workflow

```
Resume Files + Job Description
            │
            ▼
     Document Ingestion (PDF/DOCX/TXT)
            │
            ▼
     Language Detection (langdetect)
            │
            ▼
      Text Preprocessing (PII strip, normalize)
            │
   ┌────────┴────────┐
   ▼                 ▼
Skill Extraction   SBERT Embeddings
   │                 │
   ▼                 ▼
Skill Overlap   Semantic Similarity
   └────────┬────────┘
            ▼
      Blended Scoring (0.7 semantic + 0.3 skill)
            │
            ▼
      Resume Ranking
            │
            ▼
     Active-Learning Reranker (if trained)
            │
            ▼
     Suitability Flag (threshold-based)
            │
            ▼
     Explainability Layer
            │
            ▼
     Upskilling Recommendations [NEW]
            │
            ▼
     Dashboard / API Output
```

---

## 🧠 ML / AI Pipeline Step-by-Step

### 1. Ingestion

The system reads resumes from **PDF**, **DOCX**, and **TXT** formats using:

- `pdfplumber` — robust PDF text extraction
- `python-docx` — DOCX paragraph extraction
- Built-in `pathlib` — TXT file reading

```python
from src.ingestion import read_document, read_folder

text = read_document("resume.pdf")
resumes = read_folder("data/raw/resumes/")  # {filename: text}
```

### 2. Preprocessing

The preprocessing pipeline:

- **Removes emails** — regex `\S+@\S+`
- **Removes phone numbers** — regex `\+?\d[\d\-\s()]{7,}\d`
- **Removes URLs** — regex `https?://\S+|www\.\S+`
- **Normalizes whitespace** — collapse multiple spaces/newlines
- **Tokenizes** — spaCy lemmatization with stopword/punctuation removal

> **Design choice:** PII is stripped before any downstream processing to ensure privacy and prevent bias from candidate names/contact info leaking into embeddings.

### 3. Language Detection

`langdetect` (Google's language-detection library, pure-Python port) identifies:

- Resume language
- Job description language

```python
from src.language import detect_language

lang = detect_language(resume_text)  # e.g., "de", "fr", "es", "en"
```

> **Design choice:** No external translation APIs are used. Resumes are sensitive documents; the pipeline stays fully offline. Detection is deterministic (`DetectorFactory.seed = 0`) for reproducible evaluation.

### 4. Skill Extraction

A **spaCy PhraseMatcher** extracts canonical skills using aliases, abbreviations, and synonyms from `data/skills_db.json`.

```python
from src.skills import SkillExtractor, skill_overlap

extractor = SkillExtractor("data/skills_db.json")
jd_skills = extractor.extract(jd_text)
resume_skills = extractor.extract(resume_text)
score, matched, missing = skill_overlap(resume_skills, jd_skills)
```

> **Multilingual adaptation:** The skill extractor switched from `en_core_web_sm` to `spacy.blank("xx")` (language-agnostic tokenizer). This preserves exact-phrase matching for language-invariant tokens (Python, Docker, AWS, SQL) across all languages while the multilingual semantic score covers translated concepts.

### 5. Embedding Generation

SBERT embeddings are generated using **`paraphrase-multilingual-MiniLM-L12-v2`**:

- **L2 normalized** — dot product equals cosine similarity
- **Batch encoding** — configurable batch size for throughput
- **Lazy singleton** — model loaded once and cached globally

```python
from src.embeddings import embed, embed_one

jd_vec = embed_one(jd_text)
resume_vecs = embed(resume_texts)  # batch encode
similarities = resume_vecs @ jd_vec  # cosine similarity
```

> **Design choice:** The multilingual model was chosen over the monolingual `all-MiniLM-L6-v2` to enable cross-lingual matching without translation pipelines. The monolingual model is kept in config as a fallback option.

### 6. Matching Logic

The ranking score is a **weighted blend**:

```
Final Score = 0.7 × Semantic Similarity + 0.3 × Skill Overlap
```

- **Semantic similarity** (70%) captures contextual and conceptual alignment between resume and JD — crucial for understanding paraphrased requirements and implied experience.
- **Skill overlap** (30%) provides grounded, explainable evidence of explicit requirement coverage.

> **Design choice:** The 70/30 split was chosen because semantic similarity generalizes better across phrasing variations, while skill overlap prevents false positives from generic resumes that "sound right" but lack concrete qualifications.

### 7. Active Learning Integration

When the `ActiveLearningRanker` is trained, the pipeline enters **adaptive mode**:

1. **Feature extraction** — 6 features per candidate: semantic score, skill score, matched count, missing count, cross-lingual flag, recruiter preference score
2. **Logistic regression prediction** — `P(suitable)` in [0, 1]
3. **Re-sorting** — candidates reordered by adaptive score
4. **Uncertainty sampling** — top-5 most uncertain candidates flagged for recruiter review

```python
from src.matching import ResumeMatcher
from src.active_learning import load_or_train_from_disk

matcher = ResumeMatcher()
ranker = load_or_train_from_disk()
results = matcher.rank_with_feedback(jd_text, resumes, ranker=ranker)
```

> **Cold-start safety:** If fewer than `MIN_FEEDBACK_TO_TRAIN` (default 10) samples exist, or if both classes aren't represented, the system falls back to static blended scoring. This prevents unstable early predictions.

### 8. Suitability Classification

Candidates are classified using an **adaptive threshold**:

- If adaptive score exists: `adaptive_score >= ADAPTIVE_THRESHOLD` (default 0.50)
- Otherwise: `final_score >= MATCH_THRESHOLD` (default 0.50)

> **Threshold tuning:** `evaluation/sweep_threshold.py` performs a grid search to find the F1-optimal threshold on the evaluation corpus.

### 9. Explainability Layer

The system generates a structured explanation for every candidate:

```python
from src.explain import explain

info = explain(resume_text, matched_skills, missing_skills, extractor)
# {
#   "evidence": {"python": "Built ML pipelines with Python and TensorFlow."},
#   "structured": StructuredResume(experience=[...], education=[...], certifications=[...]),
#   "recommendations": [{"skill": "Kubernetes", "demand_score": 0.95, "roadmap": [...]}]
# }
```

- **Evidence sentences** — the first resume sentence mentioning each matched skill
- **Structured sections** — parsed experience, education, and certification entries
- **Missing skills** — explicit gap list for recruiter review

### 10. Recommendation Engine [NEW]

Missing skills are mapped against **market demand** and **curated learning roadmaps** from `data/market_trends.json`:

```python
from src.recommendations import SkillRecommender

recommender = SkillRecommender()
roadmaps = recommender.generate_roadmap(missing_skills, top_n=3)
```

**How it works:**

1. **Check explicit data** — For each missing skill, look up `market_trends.json` for a curated roadmap with demand score
2. **Smart generic fallback** — If no explicit data exists, generate a default roadmap:
   - Step 1: Review official documentation
   - Step 2: Build a mini-project
3. **Sort by priority** — All recommendations sorted by `demand_score` descending (highest value skills ranked first)
4. **Return top-N** — Configurable number of recommendations per candidate

**Example Output:**

```python
[
    {
        "skill": "Kubernetes",
        "demand_score": 0.95,
        "roadmap": [
            {"step": 1, "action": "Complete the Kubernetes Basics course.", "resource": "K8s Official Docs"},
            {"step": 2, "action": "Deploy a microservice app on a local cluster.", "resource": "Minikube Tutorial"},
            {"step": 3, "action": "Study Helm charts and CI/CD integration.", "resource": "Helm Documentation"}
        ]
    },
    {
        "skill": "Rust",
        "demand_score": 0.30,
        "roadmap": [
            {"step": 1, "action": "Review the official documentation and getting-started guides for Rust.", "resource": "Official Rust Docs"},
            {"step": 2, "action": "Build a small standalone mini-project to practice core concepts of Rust.", "resource": "Self-Directed Project"}
        ]
    }
]
```

> **Design choice:** The recommendation engine uses a **static market trends datastore** (`market_trends.json`) rather than live APIs, ensuring the system remains fully offline and privacy-compliant. The generic fallback guarantees every missing skill receives actionable guidance even without curated data.

---

## 🧪 Active Learning Implementation

### Strategy: Uncertainty Sampling

The active learning loop follows a **query-by-committee** style uncertainty sampling strategy:

1. **Score all candidates** with the current model (static or adaptive)
2. **Measure uncertainty** as distance from the decision boundary: `uncertainty = 1 - min(1, 2 × |score - threshold|)`
3. **Surface top-K uncertain candidates** in the review queue
4. **Recruiter labels** them (shortlist / reject)
5. **Retrain** the logistic regression reranker
6. **Repeat**

### Learning Model: Logistic Regression Reranker

The `ActiveLearningRanker` is an **L2-regularized logistic regression** classifier over 6 interpretable features:

| Feature | Description |
|---------|-------------|
| `semantic_score` | SBERT cosine similarity |
| `skill_score` | Explicit skill overlap ratio |
| `num_matched` | Count of matched skills |
| `num_missing` | Count of missing skills |
| `cross_lingual` | 1.0 if resume language ≠ JD language |
| `preference_score` | Laplace-smoothed recruiter preference for matched skills |

**Why Logistic Regression?**

- **Interpretable coefficients** — directly readable feature weights satisfy explainability requirements
- **Low-data friendly** — regularization (`C=0.5`) prevents overfitting on sparse recruiter feedback
- **Lightweight** — sub-millisecond inference, no GPU required
- **Additive design** — falls back gracefully to static scoring when untrained

### Cold Start Handling

```python
# config.py
ENABLE_ACTIVE_LEARNING = True
MIN_FEEDBACK_TO_TRAIN = 10      # minimum samples before first training
MIN_PER_CLASS_TO_TRAIN = 2      # need both positives and negatives
LR_REGULARIZATION_C = 0.5       # strong regularization for sparse data
REVIEW_QUEUE_SIZE = 5           # candidates to surface per round
```

Until these thresholds are met, the pipeline uses the static `SEMANTIC_WEIGHT / SKILL_WEIGHT` blend with no behavioral change for existing callers.

### Simulated Evaluation

`evaluation/active_learning_eval.py` implements a **simulated-oracle harness**:

- Ground-truth labels from `data/eval/labels.csv` act as the "recruiter"
- Pool/test split (80/20) ensures honest evaluation
- Learning curve tracks Precision, Recall, F1, MRR, NDCG as feedback accumulates
- Results written to `evaluation/active_learning_curve.csv`

---

## 🌍 Multilingual Support

The multilingual extension enables **cross-lingual resume matching** without translation:

| Scenario | Example |
|----------|---------|
| English JD ↔ German resume | "KI-Ingenieur" matched against "AI/ML Engineer" JD |
| English JD ↔ French resume | "Développeur backend Python" matched correctly |
| English JD ↔ Spanish resume | "Analista de datos" ranked top for Data Analyst JD |

### How It Works

1. **Language detection** — `langdetect` identifies resume and JD languages
2. **Multilingual embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` maps all languages into a shared 384-dim vector space
3. **Cross-lingual scoring** — cosine similarity computed directly between embeddings of different languages
4. **Language-agnostic skill matching** — `spacy.blank("xx")` tokenizer handles skill aliases in any script

### Evaluation

`evaluation/multilingual_eval_data.py` provides a self-contained test set:

- 3 roles × 2 languages = 6 resumes
- 3 English JDs
- Strict relevance: matching-role resume in any language = relevant

Results show small parity gaps between English and translated resumes, confirming the multilingual pipeline's effectiveness.

---

## 📊 Evaluation Results

### Core Evaluation (60-resume benchmark corpus)

| Metric | Score |
|--------|-------|
| Precision | 0.902 |
| Recall | 0.767 |
| F1 Score | 0.829 |
| MRR | 1.000 |
| NDCG | 0.986 |
| Avg Speed | ~0.1s / resume |

### Robustness Tests

| Test | Rank Drift |
|------|-----------|
| Resume truncation | +0.17 |
| Typo injection | +0.00 |
| Remove skills section | +8.50 |

**Key Finding:** The model is highly robust to typos and partial truncation but sensitive to missing skill sections (as expected, since 30% of the score depends on explicit skill overlap).

### Multilingual Evaluation

| Scenario | Parity Gap |
|----------|-----------|
| English ↔ German | 0.047 |
| English ↔ French | 0.084 |
| English ↔ Spanish | 0.017 |

The multilingual pipeline successfully matched translated resumes directly against English job descriptions with minimal rank drift.

### Active Learning Learning Curve

See `evaluation/active_learning_curve.csv` for round-by-round metrics. The simulated-oracle harness demonstrates consistent improvement in F1 and NDCG as recruiter feedback accumulates.

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- ~2GB free disk space (for SBERT model download)

### 1. Clone Repository

```bash
git clone https://github.com/devn-cmd/Resume-Matcher.git
cd Resume-Matcher
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` is organized into clean sections:

| Category | Packages |
|----------|----------|
| Web UI | `streamlit==1.57.0` |
| Data | `numpy==2.4.6`, `pandas==3.0.3` |
| Embeddings / NLP | `sentence-transformers==5.5.1`, `spacy==3.8.14`, `langdetect==1.0.9` |
| spaCy English model | `en_core_web_sm` (direct wheel URL) |
| Document ingestion | `pdfplumber==0.11.9`, `python-docx==1.2.0` |
| Active learning | `scikit-learn==1.8.0`, `joblib==1.5.3` |
| API service | `fastapi==0.136.3`, `uvicorn==0.47.0`, `python-multipart==0.0.29` |
| Tests | `pytest==9.0.3` |

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

> Note: The `en_core_web_sm` model is used by the section parser (`src/sections.py`) and preprocessing pipeline. The skill extractor uses `spacy.blank("xx")` and does not require this download.

---

## ▶️ Usage Instructions

### Run FastAPI Server

```bash
uvicorn api.main:app --reload
```

Open the interactive docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Available endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info & endpoint listing |
| `/rank` | POST | Rank resume strings against a JD (JSON) |
| `/rank-files` | POST | Rank uploaded resume files (multipart) |
| `/feedback` | POST | Log one recruiter judgement |
| `/retrain` | POST | Retrain adaptive reranker on accumulated feedback |
| `/active-learning/status` | GET | Inspect feedback counts, model state, coefficients |

### Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Run Tests

```bash
pytest
```

### Run Evaluation

```bash
# Core evaluation
python -m evaluation.run_eval

# Threshold sweep
python -m evaluation.sweep_threshold

# Robustness tests
python -m evaluation.robustness

# Active learning learning curve
python -m evaluation.active_learning_eval --rounds 12 --k 5 --plot

# Multilingual evaluation
python -m evaluation.run_multilingual_eval
```

### Run Colab Demo

Open `resume_matcher_demo.ipynb` in Google Colab or Jupyter. The notebook walks through:

1. Environment setup
2. Model loading
3. Resume ingestion
4. Ranking and explanation
5. Active learning simulation
6. Multilingual matching demonstration
7. Upskilling recommendation generation

---

## 🧾 Example Inputs and Outputs

### Example Job Description

```text
Looking for a Python Machine Learning Engineer with NLP and Docker experience.
```

### Example Resume Skills

```text
Python, NLP, Deep Learning, Docker, SQL
```

### API Response

```json
{
  "resume_id": "resume_01.pdf",
  "final_score": 0.91,
  "semantic_score": 0.88,
  "skill_score": 1.0,
  "suitable": true,
  "matched_skills": ["python", "docker", "nlp"],
  "missing_skills": [],
  "language": "en",
  "language_name": "English",
  "jd_language": "en",
  "cross_lingual": false,
  "adaptive_score": null,
  "recruiter_preference_score": 0.5,
  "is_uncertain": false
}
```

### Dashboard Output

| Candidate | Language | Static Score | Adaptive | Semantic | Skill Match | Suitable |
|-----------|----------|-------------|----------|----------|-------------|----------|
| resume_01.pdf | English | 91% | — | 88% | 100% | ✅ |
| resume_02.pdf | German 🌐 | 87% | — | 85% | 95% | ✅ |

### Upskilling Recommendation Output

```json
{
  "recommendations": [
    {
      "skill": "Kubernetes",
      "demand_score": 0.95,
      "roadmap": [
        {"step": 1, "action": "Complete the Kubernetes Basics course.", "resource": "K8s Official Docs"},
        {"step": 2, "action": "Deploy a microservice app on a local cluster.", "resource": "Minikube Tutorial"},
        {"step": 3, "action": "Study Helm charts and CI/CD integration.", "resource": "Helm Documentation"}
      ]
    },
    {
      "skill": "Rust",
      "demand_score": 0.30,
      "roadmap": [
        {"step": 1, "action": "Review the official documentation and getting-started guides for Rust.", "resource": "Official Rust Docs"},
        {"step": 2, "action": "Build a small standalone mini-project to practice core concepts of Rust.", "resource": "Self-Directed Project"}
      ]
    }
  ]
}
```

---

## 🧩 Important Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **SBERT embeddings** | Strong semantic understanding; pre-trained on paraphrase tasks |
| **Hybrid scoring (70/30)** | Combines contextual meaning with grounded, explainable skill evidence |
| **Logistic regression reranker** | Interpretable coefficients; low-data friendly; lightweight inference |
| **spaCy PhraseMatcher** | Reliable deterministic skill extraction with synonym support |
| **Multilingual embeddings** | Cross-language compatibility without translation APIs (privacy-safe) |
| **Feature branches** | Clean engineering workflow; each stretch goal isolated and mergeable |
| **Cached models** | `@st.cache_resource` and global singletons for sub-second dashboard responsiveness |
| **Append-only JSONL feedback** | Crash-safe, auditable, privacy-compliant (no raw resume text stored) |
| **Language-agnostic tokenizer** | `spacy.blank("xx")` enables skill matching across scripts without model downloads |
| **Static market trends datastore** | Offline, privacy-compliant; no external API calls for recommendations |
| **Smart generic fallback** | Every missing skill gets actionable guidance, even without curated data |

---

## 🌱 Git Workflow & Project Evolution

The project followed an **incremental engineering workflow** with feature branches for each major capability.

### Main Development Milestones

| Phase | Commits | Features |
|-------|---------|----------|
| Core Pipeline | Initial commits | Ingestion, preprocessing, skill extraction, embeddings, ranking, evaluation, API service |
| Explainability | `feature/explainability` → merge | Matched/missing skills, evidence sentences, structured sections, CSV export with Yes/No suitability |
| Skill Recommendations | `feature/skill-recommendations` → merge | Missing skill detection, market trends data, personalized upskilling roadmaps |
| Multilingual Support | `feature/multilingual-support` → merge | Language detection, multilingual SBERT model, language-agnostic tokenizer, cross-lingual evaluation |
| Active Learning | `feature/active-learning` → merge | Feedback store, uncertainty sampling, LR reranker, review queue, retrain endpoints, learning curve harness |
| Polish & Demo | Final commits | Cleaned requirements.txt, Colab demo notebook, seeded evaluation data |

### Branch-Based Development

Each stretch goal was developed on a dedicated feature branch and merged into `main` using `git merge --no-ff` to preserve project history:

```bash
# Example workflow
git checkout -b feature/explainability
# ... develop explainability features ...
git commit -m "feat: add matched/missing skills with evidence"
git checkout main
git merge --no-ff feature/explainability
```

### Key Commits

| Commit | Message | Significance |
|--------|---------|------------|
| `0809baa` | `Merge feature/explainability` | First stretch goal: explainable scoring |
| `a75cfcf` | `Merge feature/skill-recommendations` | Second stretch goal: upskilling roadmaps |
| `732f470` | `Merge feature/multilingual-support` | Third stretch goal: cross-lingual matching |
| `b4d6bf8` | `Merge feature/active-learning` | Fourth stretch goal: feedback-driven learning |
| `6a0ee77` | `data: add eval resume corpus` | Seeded 60-resume benchmark for reproducibility |
| `40f0431` | `docs: add end-to-end Colab demo notebook` | Interactive documentation |
| `75360ec` | `chore: clean requirements.txt` | Production-ready dependency list |

---

## 🚧 Challenges Solved

### Dependency Conflicts

**Problem:** NumPy ABI incompatibilities, torchvision import issues, and transformers version conflicts caused installation failures.

**Solution:** Pinned stable ML package versions in `requirements.txt` with explicit version constraints. Removed heavy transitive dependencies (e.g., `torchvision`, `transformers`) where possible, relying on `sentence-transformers` to pull a compatible `torch` automatically.

### Semantic Similarity Floor

**Problem:** SBERT assigned moderate similarity (~0.5) to unrelated resumes, causing false positives.

**Solution:** Introduced explicit skill overlap scoring (30% weight) and tuned the match threshold using F1 optimization on the evaluation corpus. The hybrid approach grounds semantic similarity with concrete requirement evidence.

### Streamlit Performance

**Problem:** Model reloads on every dashboard interaction caused 5–10 second delays.

**Solution:** Implemented `@st.cache_resource` decorators for the `ResumeMatcher`, `SkillExtractor`, and `ActiveLearningRanker`. The SBERT model is warmed up once on startup via `embed_one("warmup")`.

### Skill Match Saturation

**Problem:** Generic skills (e.g., "communication") caused inflated overlap scores.

**Solution:** Expanded the curated `skills_db.json` with domain-specific aliases and weighted scoring toward technical skills. The semantic score (70%) dominates, preventing generic skill inflation from overwhelming contextual relevance.

### Cross-Lingual Skill Matching

**Problem:** English-only spaCy model failed to tokenize German/French/Spanish skill phrases.

**Solution:** Replaced `en_core_web_sm` with `spacy.blank("xx")` in the skill extractor. This language-agnostic tokenizer handles any script while preserving exact-phrase matching for language-invariant technical terms.

### Active Learning Cold Start

**Problem:** Early recruiter feedback is sparse and imbalanced, leading to unstable model training.

**Solution:** Implemented strict cold-start gating (`MIN_FEEDBACK_TO_TRAIN=10`, `MIN_PER_CLASS_TO_TRAIN=2`) with strong L2 regularization (`C=0.5`). The system falls back to static scoring until sufficient balanced feedback is collected.

### Recommendation Data Scarcity

**Problem:** Not all skills have curated roadmap data in `market_trends.json`.

**Solution:** Implemented a **smart generic fallback** in `SkillRecommender.generate_roadmap()` that auto-generates a two-step learning path (official docs → mini-project) for any unrecognized skill. This ensures every candidate receives actionable upskilling guidance regardless of data coverage.



## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **NLP** | spaCy, Sentence Transformers (SBERT), langdetect |
| **ML** | scikit-learn (LogisticRegression), joblib |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Document Parsing** | pdfplumber, python-docx |
| **Data** | NumPy, Pandas |
| **Testing** | pytest |
| **Language Detection** | langdetect |
| **Version Control** | Git with feature-branch workflow |



## 👨‍💻 Author

Built as an end-to-end NLP engineering project demonstrating:

- Semantic search & dense retrieval
- Information extraction & structured parsing
- Explainable AI with interpretable models
- Multilingual NLP without translation
- Evaluation engineering & benchmarking
- Active learning systems
- Production-oriented ML architecture
- Market-driven recommendation systems

**Repository:** [https://github.com/devn-cmd/Resume-Matcher](https://github.com/devn-cmd/Resume-Matcher)
