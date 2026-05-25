from dataclasses import dataclass, field
import config
from src.preprocessing import preprocess
from src.embeddings import embed, embed_one
from src.skills import SkillExtractor, skill_overlap


@dataclass
class MatchResult:
    resume_id: str
    final_score: float
    semantic_score: float
    skill_score: float
    matched_skills: set = field(default_factory=set)
    missing_skills: set = field(default_factory=set)


class ResumeMatcher:
    def __init__(self):
        self.extractor = SkillExtractor(config.SKILLS_DB_PATH)

    def rank(self, jd_text: str, resumes: dict[str, str]) -> list[MatchResult]:
        ids = list(resumes.keys())
        resume_texts = [preprocess(resumes[i])["clean"] for i in ids]
        jd_clean = preprocess(jd_text)["clean"]

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
            results.append(MatchResult(
                resume_id=rid,
                final_score=round(final, 4),
                semantic_score=round(float(semantic[idx]), 4),
                skill_score=round(sk_score, 4),
                matched_skills=matched,
                missing_skills=missing,
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

    def is_suitable(self, result: MatchResult) -> bool:
        return result.final_score >= config.MATCH_THRESHOLD