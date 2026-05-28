
from dataclasses import dataclass, field
from typing import Optional
import config
from src.preprocessing import preprocess
from src.embeddings import embed, embed_one
from src.skills import SkillExtractor, skill_overlap
from src.language import detect_language


@dataclass
class MatchResult:
    resume_id: str
    final_score: float
    semantic_score: float
    skill_score: float
    matched_skills: set = field(default_factory=set)
    missing_skills: set = field(default_factory=set)
    language: str = config.DEFAULT_LANGUAGE       # detected résumé language
    jd_language: str = config.DEFAULT_LANGUAGE     # detected JD language
    cross_lingual: bool = False                    # résumé lang != JD lang
# --- Active-learning fields (optional; populated by apply_active_learning) ---
# All default to None / safe values so existing callers see no behavioural change.

    adaptive_score: Optional[float] = None
    recruiter_preference_score: float = 0.5
    is_uncertain: bool = False

class ResumeMatcher:
    def __init__(self):
        self.extractor = SkillExtractor(config.SKILLS_DB_PATH)

    def rank(self, jd_text: str, resumes: dict[str, str]) -> list[MatchResult]:
        ids = list(resumes.keys())
        resume_texts = [preprocess(resumes[i])["clean"] for i in ids]
        jd_clean = preprocess(jd_text)["clean"]

        # Detect languages on the raw text (before PII stripping mangles it).
        jd_lang = (detect_language(jd_text)
                   if config.ENABLE_LANGUAGE_DETECTION else config.DEFAULT_LANGUAGE)

        jd_vec = embed_one(jd_clean)
        resume_vecs = embed(resume_texts)
        semantic = resume_vecs @ jd_vec
        semantic = (semantic + 1) / 2          # map [-1,1] -> [0,1]

        jd_skills = self.extractor.extract(jd_text)

        results = []
        for idx, rid in enumerate(ids):
            r_skills = self.extractor.extract(resumes[rid])
            sk_score, matched, missing = skill_overlap(r_skills, jd_skills)
            final = (config.SEMANTIC_WEIGHT * float(semantic[idx])
                     + config.SKILL_WEIGHT * sk_score)
            r_lang = (detect_language(resumes[rid])
                      if config.ENABLE_LANGUAGE_DETECTION else config.DEFAULT_LANGUAGE)
            results.append(MatchResult(
                resume_id=rid,
                final_score=round(final, 4),
                semantic_score=round(float(semantic[idx]), 4),
                skill_score=round(sk_score, 4),
                matched_skills=matched,
                missing_skills=missing,
                language=r_lang,
                jd_language=jd_lang,
                cross_lingual=(r_lang != jd_lang),
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results
    # -------- Active-learning aware ranking --------
    def rank_with_feedback(self, jd_text: str, resumes: dict[str, str], ranker=None) -> list[MatchResult]:
        """Static rank + optional re-scoring by a trained ActiveLearningRanker.
 
        Backward-compatible: if `ranker` is None or untrained, this returns the
        exact output of `rank()`. When trained, it attaches `adaptive_score` and
        re-sorts the results. The static fields (`final_score`, etc.) are kept
        intact so explainability still has the un-blended view.
        """
        from src.active_learning import apply_active_learning, select_review_queue
        results = self.rank(jd_text, resumes)
        results = apply_active_learning(results, ranker)
        # Flag the most uncertain candidates for the review queue (UI / API surface).
        select_review_queue(results, ranker)
        return results
    


    def is_suitable(self, result: MatchResult) -> bool:
        """Uses adaptive threshold when re-ranker has produced a probability,
        falls back to the static threshold otherwise — same behaviour as before
        for any path that hasn't opted into active learning."""
        if result.adaptive_score is not None:
            return result.adaptive_score >= config.ADAPTIVE_THRESHOLD
        return result.final_score >= config.MATCH_THRESHOLD