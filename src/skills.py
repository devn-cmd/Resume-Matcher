# # src/skills.py
# import json
# import spacy
# from spacy.matcher import PhraseMatcher

# _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# class SkillExtractor:
#     def __init__(self, skills_db_path: str):
#         with open(skills_db_path) as f:
#             self.db = json.load(f)               # canonical -> [synonyms]
#         self.matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
#         self.alias_to_canonical = {}
#         for canonical, aliases in self.db.items():
#             patterns = [_nlp.make_doc(a) for a in aliases]
#             self.matcher.add(canonical, patterns)
#             for a in aliases:
#                 self.alias_to_canonical[a.lower()] = canonical

#     def extract(self, text: str) -> set[str]:
#         """Return the set of canonical skills found in text."""
#         doc = _nlp(text.lower())
#         found = set()
#         for match_id, start, end in self.matcher(doc):
#             found.add(_nlp.vocab.strings[match_id])
#         return found


# def skill_overlap(resume_skills: set, jd_skills: set) -> tuple[float, set, set]:
#     """Coverage of JD skills + matched/missing for explainability."""
#     if not jd_skills:
#         return 0.0, set(), set()
#     matched = resume_skills & jd_skills
#     missing = jd_skills - resume_skills
#     score = len(matched) / len(jd_skills)
#     return score, matched, missing


import json
import spacy
from spacy.matcher import PhraseMatcher

# MULTILINGUAL CHANGE: use spaCy's language-agnostic "MultiLanguage" tokenizer
# (spacy.blank("xx")) instead of the English-only `en_core_web_sm`.
#
# Why this is safe: extract() below only ever used the *tokenizer* (it disabled
# the parser/NER and never touched lemmas or POS), so swapping in a universal
# tokenizer changes nothing for English while making the matcher work for any
# language. It also drops the hard dependency on the English model in this path
# and needs no model download.
_nlp = spacy.blank("xx")


class SkillExtractor:
    def __init__(self, skills_db_path: str):
        with open(skills_db_path, encoding="utf-8") as f:
            self.db = json.load(f)               # canonical -> [synonyms]
        self.matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
        self.alias_to_canonical = {}
        for canonical, aliases in self.db.items():
            patterns = [_nlp.make_doc(a) for a in aliases]
            self.matcher.add(canonical, patterns)
            for a in aliases:
                self.alias_to_canonical[a.lower()] = canonical

    def extract(self, text: str) -> set[str]:
        """Return the set of canonical skills found in text.

        Matching is exact-phrase over lowercased tokens, so it catches
        language-invariant tokens (Python, Docker, AWS, SQL...) in any language
        automatically, plus any multilingual aliases present in skills_db.json.
        Concept skills that get *translated and inflected* (e.g. German
        "maschinellem Lernen") may be missed here — that gap is covered by the
        multilingual semantic score in matching.py, which carries 70% of the
        weight.
        """
        doc = _nlp(text.lower())
        found = set()
        for match_id, start, end in self.matcher(doc):
            found.add(_nlp.vocab.strings[match_id])
        return found


def skill_overlap(resume_skills: set, jd_skills: set) -> tuple[float, set, set]:
    """Coverage of JD skills + matched/missing for explainability."""
    if not jd_skills:
        return 0.0, set(), set()
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    score = len(matched) / len(jd_skills)
    return score, matched, missing