import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from src.matching import ResumeMatcher


def evaluate(jd_texts: dict, resumes: dict, labels: pd.DataFrame):
    matcher = ResumeMatcher()
    y_true_cls, y_pred_cls = [], []
    rr_list, ndcg_list = [], []
    matched_sims, unmatched_sims = [], []
    times = []

    for jd_id, jd_text in jd_texts.items():
        rel = labels[labels.jd_id == jd_id]
        rel_map = dict(zip(rel.resume_id, rel.relevant))
        subset = {rid: resumes[rid] for rid in rel_map if rid in resumes}
        if not subset:
            continue

        t0 = time.perf_counter()
        ranked = matcher.rank(jd_text, subset)
        times.append((time.perf_counter() - t0) / len(subset))

        for r in ranked:
            truth = rel_map.get(r.resume_id, 0)
            y_true_cls.append(truth)
            y_pred_cls.append(int(matcher.is_suitable(r)))
            (matched_sims if truth == 1 else unmatched_sims).append(r.semantic_score)

        order = [r.resume_id for r in ranked]
        first_rel = next((i + 1 for i, rid in enumerate(order)
                          if rel_map.get(rid, 0) == 1), None)
        if first_rel:
            rr_list.append(1.0 / first_rel)
        true_rel = np.array([[rel_map.get(rid, 0) for rid in order]])
        pred_sc = np.array([[r.final_score for r in ranked]])
        if true_rel.sum() > 0 and true_rel.shape[1] > 1:
            ndcg_list.append(ndcg_score(true_rel, pred_sc))

    return {
        "precision": round(precision_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        "recall":    round(recall_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        "f1":        round(f1_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        "mrr":       round(float(np.mean(rr_list)) if rr_list else 0.0, 4),
        "ndcg":      round(float(np.mean(ndcg_list)) if ndcg_list else 0.0, 4),
        "cosine_matched_mean":   round(float(np.mean(matched_sims)) if matched_sims else 0.0, 4),
        "cosine_unmatched_mean": round(float(np.mean(unmatched_sims)) if unmatched_sims else 0.0, 4),
        "avg_seconds_per_resume": round(float(np.mean(times)) if times else 0.0, 5),
    }