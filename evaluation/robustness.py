"""
Robustness check (brief requirement).

Takes the resumes that are TRULY relevant to one job description, damages them
three ways (truncate to 30%, inject typos, drop the skills section), re-ranks,
and reports how far they drift from their original rank/score.

A robust system degrades gracefully: scores dip, but relevant resumes stay
near the top instead of collapsing.

Run from project root:
    python -m evaluation.robustness
"""
import random
import numpy as np
import pandas as pd
from src.matching import ResumeMatcher


def _load_dict(path, k, v):
    df = pd.read_csv(path)
    return dict(zip(df[k].astype(str), df[v].astype(str)))


# ---- damage functions --------------------------------------------------------
def truncate(text, frac=0.30):
    """Keep only the first 30% of the words."""
    words = text.split()
    return " ".join(words[: max(1, int(len(words) * frac))])


def inject_typos(text, rate=0.10, seed=0):
    """Drop a character from ~10% of longer words."""
    rng = random.Random(seed)
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(len(w))
            w = w[:i] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def drop_skills(text):
    """Remove lines belonging to a 'skills' section."""
    out, skip = [], False
    for ln in text.splitlines():
        low = ln.strip().lower()
        if low.startswith("skill"):
            skip = True
            continue
        if skip and any(low.startswith(h) for h in
                        ("education", "experience", "company", "project")):
            skip = False
        if not skip:
            out.append(ln)
    return "\n".join(out)


DAMAGES = [
    ("truncate to 30%", truncate),
    ("inject typos", inject_typos),
    ("drop skills section", drop_skills),
]


def main():
    resumes = _load_dict("data/eval/resumes.csv", "resume_id", "text")
    jd_texts = _load_dict("data/eval/jds.csv", "jd_id", "text")
    labels = pd.read_csv("data/eval/labels.csv")

    # Use the first job description.
    jd_id, jd_text = next(iter(jd_texts.items()))
    rel = labels[labels.jd_id == jd_id]
    rel_map = dict(zip(rel.resume_id.astype(str), rel.relevant))
    subset = {rid: resumes[rid] for rid in rel_map if rid in resumes}
    relevant_ids = [rid for rid, v in rel_map.items() if int(v) == 1 and rid in resumes]

    matcher = ResumeMatcher()

    # Baseline ranking (no damage).
    base = matcher.rank(jd_text, subset)
    base_rank = {r.resume_id: i + 1 for i, r in enumerate(base)}
    base_score = {r.resume_id: r.final_score for r in base}

    print(f"Job description: {jd_id}")
    print(f"Relevant resumes: {len(relevant_ids)} of {len(subset)} total\n")
    print(f"Baseline: relevant resumes occupy ranks "
          f"{sorted(base_rank[r] for r in relevant_ids)}\n")

    print(f"{'damage':>22}{'avg rank drift':>16}{'avg score drop':>16}")
    for name, fn in DAMAGES:
        damaged = {rid: (fn(text) if rid in relevant_ids else text)
                   for rid, text in subset.items()}
        ranked = matcher.rank(jd_text, damaged)
        new_rank = {r.resume_id: i + 1 for i, r in enumerate(ranked)}
        new_score = {r.resume_id: r.final_score for r in ranked}

        drift = np.mean([new_rank[r] - base_rank[r] for r in relevant_ids])
        drop = np.mean([base_score[r] - new_score[r] for r in relevant_ids])
        print(f"{name:>22}{drift:>+16.2f}{drop:>+16.4f}")

    print("\n(rank drift = positions moved down; score drop = points lost. "
          "Small numbers = robust.)")


if __name__ == "__main__":
    main()