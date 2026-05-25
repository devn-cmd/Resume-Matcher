"""
Finds the MATCH_THRESHOLD that maximizes F1.
Embeds the resumes once, then tests several cutoffs.

Run from project root:
    python -m evaluation.sweep_threshold
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.matching import ResumeMatcher


def _load_dict(path, k, v):
    df = pd.read_csv(path)
    return dict(zip(df[k].astype(str), df[v].astype(str)))


def main():
    resumes = _load_dict("data/eval/resumes.csv", "resume_id", "text")
    jd_texts = _load_dict("data/eval/jds.csv", "jd_id", "text")
    labels = pd.read_csv("data/eval/labels.csv")

    matcher = ResumeMatcher()
    scores, truths = [], []
    for jd_id, jd_text in jd_texts.items():
        rel = labels[labels.jd_id == jd_id]
        rel_map = dict(zip(rel.resume_id, rel.relevant))
        subset = {rid: resumes[rid] for rid in rel_map if rid in resumes}
        if not subset:
            continue
        for r in matcher.rank(jd_text, subset):
            scores.append(r.final_score)
            truths.append(int(rel_map.get(r.resume_id, 0)))

    scores = np.array(scores)
    truths = np.array(truths)

    print(f"{'threshold':>10}{'precision':>11}{'recall':>9}{'f1':>9}")
    best_t, best_f1 = None, -1.0
    for t in np.round(np.arange(0.30, 0.66, 0.05), 2):
        pred = (scores >= t).astype(int)
        p = precision_score(truths, pred, zero_division=0)
        rc = recall_score(truths, pred, zero_division=0)
        f = f1_score(truths, pred, zero_division=0)
        print(f"{t:>10.2f}{p:>11.4f}{rc:>9.4f}{f:>9.4f}")
        if f > best_f1:
            best_t, best_f1 = t, f

    print(f"\nBest F1 = {best_f1:.4f} at threshold = {best_t:.2f}")
    print(f"-> update MATCH_THRESHOLD in config.py to {best_t:.2f}")


if __name__ == "__main__":
    main()
