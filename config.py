MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MONOLINGUAL_MODEL_NAME = "all-MiniLM-L6-v2"
SKILLS_DB_PATH = "data/skills_db.json"
SEMANTIC_WEIGHT = 0.7      # weight on embedding similarity
SKILL_WEIGHT = 0.3         # weight on explicit skill overlap
MATCH_THRESHOLD = 0.50     # tuned


ENABLE_LANGUAGE_DETECTION = True
DEFAULT_LANGUAGE = "en"        # fallback when text is too short / undetectable
MIN_CHARS_FOR_DETECTION = 20   # langdetect is unreliable below this


# -----------------------------------------------------------------------------
# Active learning settings
# -----------------------------------------------------------------------------
# The active-learning layer is OFF until enough feedback has been collected;
# until then, ranking falls back to the static SEMANTIC_WEIGHT / SKILL_WEIGHT
# blend above (cold-start safety).
ENABLE_ACTIVE_LEARNING = True
FEEDBACK_PATH = "data/recruiter_feedback.jsonl"   # append-only JSONL
ACTIVE_MODEL_PATH = "data/active_model.joblib"    # trained re-ranker
MIN_FEEDBACK_TO_TRAIN = 10        # minimum samples before first training
MIN_PER_CLASS_TO_TRAIN = 2        # need both positives and negatives
LR_REGULARIZATION_C = 0.5         # small => stronger regularization (sparse data)
LR_RANDOM_SEED = 42               # reproducibility
REVIEW_QUEUE_SIZE = 5             # how many uncertain candidates to surface
ADAPTIVE_THRESHOLD = 0.50         # P(suitable) cutoff for the re-ranker