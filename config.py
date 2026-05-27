MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MONOLINGUAL_MODEL_NAME = "all-MiniLM-L6-v2"
SKILLS_DB_PATH = "data/skills_db.json"
SEMANTIC_WEIGHT = 0.7      # weight on embedding similarity
SKILL_WEIGHT = 0.3         # weight on explicit skill overlap
MATCH_THRESHOLD = 0.50     # tuned 


ENABLE_LANGUAGE_DETECTION = True
DEFAULT_LANGUAGE = "en"        # fallback when text is too short / undetectable
MIN_CHARS_FOR_DETECTION = 20   # langdetect is unreliable below this
 