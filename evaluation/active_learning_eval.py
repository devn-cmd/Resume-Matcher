"""Simulated-oracle learning-curve harness for the active-learning loop.

Why this exists
---------------
The project spec asks for "iterative" improvement from recruiter feedback. We
can't ship a live recruiter, so we *simulate* one: the existing labelled
evaluation set (jd_texts / resumes / labels) is treated as ground truth, and
the harness reveals labels to the model only as the active-learning loop queries
them. This produces a credible learning curve — Precision / Recall / F1 / MRR /
NDCG climbing as feedback accumulates — measured on a held-out test slice the
model never sees during training.

Pipeline (per round t = 1..T):
  1.  Score every (jd, resume) pair with the current model
      (static blend at t=0, adaptive once trained).
  2.  Pick the K candidates from the *pool* that the model is least sure about
      (uncertainty sampling).
  3.  "Reveal" their ground-truth labels from `labels` and log them as feedback.
  4.  Retrain the re-ranker.
  5.  Evaluate on the *test* split (never queried) — record metrics.

This mirrors how a recruiter would actually use the system: label only the few
candidates the model flags as uncertain, then watch quality improve.

Honest limitations
------------------
* The "recruiter" here is the ground-truth label set, which is consistent and
  noise-free; real recruiters disagree with each other and with themselves.
* The test split is small (depends on your label file) and the curve will be
  noisy near the end where the pool runs out.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, ndcg_score,
)

import config
from src.matching import ResumeMatcher
from src.active_learning import (
    ActiveLearningRanker, apply_active_learning, uncertainty,
    features_from_result,
)
from src.feedback import FeedbackRecord, log_batch, clear_feedback, load_feedback


# ---------------------------------------------------------------------------
# Scoring helpers — shared between rounds
# ---------------------------------------------------------------------------

def _rank_all(matcher: ResumeMatcher, ranker: ActiveLearningRanker,
              jd_texts: dict, resumes: dict, labels: pd.DataFrame):
    """Return {jd_id: [MatchResult,...]} ranked under the current model."""
    out = {}
    for jd_id, jd_text in jd_texts.items():
        rel_ids = labels.loc[labels.jd_id == jd_id, "resume_id"].tolist()
        subset = {rid: resumes[rid] for rid in rel_ids if rid in resumes}
        if not subset:
            continue
        results = matcher.rank(jd_text, subset)
        results = apply_active_learning(results, ranker)
        out[jd_id] = results
    return out


def _metrics_on_split(ranker, ranked: dict, labels_split: pd.DataFrame,
                      threshold: Optional[float] = None) -> dict:
    """P/R/F1/MRR/NDCG on a label slice. Uses adaptive threshold when trained."""
    if threshold is None:
        threshold = (config.ADAPTIVE_THRESHOLD if ranker.is_trained()
                     else config.MATCH_THRESHOLD)
    y_true, y_pred = [], []
    rr, ndcgs = [], []
    for jd_id, results in ranked.items():
        rel_map = dict(zip(
            labels_split.loc[labels_split.jd_id == jd_id, "resume_id"],
            labels_split.loc[labels_split.jd_id == jd_id, "relevant"],
        ))
        if not rel_map:
            continue
        # Classification metrics — restricted to test resumes for this JD.
        for r in results:
            if r.resume_id not in rel_map:
                continue
            score = r.adaptive_score if r.adaptive_score is not None else r.final_score
            y_true.append(int(rel_map[r.resume_id]))
            y_pred.append(int(score >= threshold))
        # Ranking metrics — use only test resumes, preserve model's ordering.
        test_in_order = [r for r in results if r.resume_id in rel_map]
        if not test_in_order:
            continue
        truths = [int(rel_map[r.resume_id]) for r in test_in_order]
        first_rel = next((i + 1 for i, t in enumerate(truths) if t == 1), None)
        if first_rel:
            rr.append(1.0 / first_rel)
        if sum(truths) > 0 and len(truths) > 1:
            scores = [
                (r.adaptive_score if r.adaptive_score is not None else r.final_score)
                for r in test_in_order
            ]
            ndcgs.append(ndcg_score(np.array([truths]), np.array([scores])))
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4) if y_true else 0.0,
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4)    if y_true else 0.0,
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4)        if y_true else 0.0,
        "mrr":       round(float(np.mean(rr)) if rr else 0.0, 4),
        "ndcg":      round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

def run_active_learning_eval(
    jd_texts: dict[str, str],
    resumes: dict[str, str],
    labels: pd.DataFrame,
    rounds: int = 12,
    queries_per_round: int = 5,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    out_csv: str | None = "evaluation/active_learning_curve.csv",
) -> pd.DataFrame:
    """Drive the simulated active-learning loop and return the learning curve.

    Parameters
    ----------
    jd_texts : {jd_id: jd_text}
    resumes  : {resume_id: resume_text}     — same shape as evaluate.py expects
    labels   : DataFrame with columns [jd_id, resume_id, relevant]
    rounds   : how many AL iterations to simulate
    queries_per_round : K — labels revealed per round (uncertainty-sampled)
    test_fraction     : held-out share never queried (honest eval slice)
    """
    rng = np.random.default_rng(random_seed)

    # Pool/test split. Stratify by JD so every JD has both pool and test rows.
    pool_rows, test_rows = [], []
    for jd_id, grp in labels.groupby("jd_id"):
        idx = rng.permutation(len(grp))
        cut = max(1, int(len(grp) * test_fraction))
        test_rows.append(grp.iloc[idx[:cut]])
        pool_rows.append(grp.iloc[idx[cut:]])
    pool   = pd.concat(pool_rows,  ignore_index=True)
    test   = pd.concat(test_rows,  ignore_index=True)
    print(f"[eval] pool={len(pool)}  test={len(test)}  "
          f"(test held out and never queried)")

    # Fresh feedback log for this run.
    clear_feedback()
    matcher = ResumeMatcher()
    ranker  = ActiveLearningRanker()

    # Baseline (round 0): static blend, no feedback.
    ranked = _rank_all(matcher, ranker, jd_texts, resumes, labels)
    curve = [{"round": 0, "feedback_count": 0, "model": "static",
              **_metrics_on_split(ranker, ranked, test)}]
    print(f"[eval] round 0 (static): {curve[-1]}")

    queried: set[tuple[str, str]] = set()
    for t in range(1, rounds + 1):
        # 1) Score everything under current model
        ranked_pool = _rank_all(matcher, ranker, jd_texts, resumes, pool)

        # 2) Uncertainty-rank candidates inside the pool, skipping already-queried.
        candidates = []
        pool_keys = set(zip(pool.jd_id, pool.resume_id))
        for jd_id, results in ranked_pool.items():
            for r in results:
                key = (jd_id, r.resume_id)
                if key in queried or key not in pool_keys:
                    continue
                candidates.append((uncertainty(r, ranker), jd_id, r))
        if not candidates:
            print(f"[eval] round {t}: pool exhausted, stopping early")
            break
        candidates.sort(key=lambda c: -c[0])      # most uncertain first
        batch = candidates[:queries_per_round]

        # 3) Reveal ground-truth labels and log as feedback.
        records = []
        for _u, jd_id, r in batch:
            row = pool[(pool.jd_id == jd_id) & (pool.resume_id == r.resume_id)]
            if row.empty:
                continue
            label = int(row.iloc[0]["relevant"])
            records.append(FeedbackRecord(
                jd_id=jd_id,
                resume_id=r.resume_id,
                semantic_score=float(r.semantic_score),
                skill_score=float(r.skill_score),
                num_matched=len(r.matched_skills),
                num_missing=len(r.missing_skills),
                cross_lingual=int(bool(r.cross_lingual)),
                matched_skills=[str(s) for s in r.matched_skills],
                action="shortlist" if label == 1 else "reject",
            ))
            queried.add((jd_id, r.resume_id))
        log_batch(records)

        # 4) Retrain on everything collected so far.
        df = load_feedback()
        report = ranker.fit(df)

        # 5) Evaluate on the held-out test slice.
        ranked_test = _rank_all(matcher, ranker, jd_texts, resumes, labels)
        metrics = _metrics_on_split(ranker, ranked_test, test)
        curve.append({
            "round": t,
            "feedback_count": len(df),
            "model": "adaptive" if report.trained else "static (warming up)",
            **metrics,
        })
        print(f"[eval] round {t}: {curve[-1]}  trained={report.trained}")

    df_curve = pd.DataFrame(curve)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_curve.to_csv(out_csv, index=False)
        print(f"[eval] learning curve written to {out_csv}")
    return df_curve


def plot_curve(df_curve: pd.DataFrame, out_png: str = "evaluation/active_learning_curve.png"):
    """Optional matplotlib plot of the learning curve. No-op (with a hint) if
    matplotlib isn't installed — the CSV is always written regardless."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless-safe
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not installed — skipping plot "
              "(`pip install matplotlib` to enable). CSV was still written.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
        ax.plot(df_curve["feedback_count"], df_curve[metric], marker="o", label=metric)
    ax.set_xlabel("Recruiter labels collected")
    ax.set_ylabel("Metric on held-out test slice")
    ax.set_title("Active-learning learning curve (simulated oracle)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"[eval] curve plotted to {out_png}")


# ---------------------------------------------------------------------------
# CLI — wire this to whatever you already use to load jd_texts / resumes / labels
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--k", type=int, default=5, help="queries per round")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Plug your existing eval fixtures in here. The harness expects the same
    # shape as evaluate.py: jd_texts={jd_id:text}, resumes={resume_id:text},
    # labels=DataFrame[jd_id, resume_id, relevant].
    try:
        from evaluation.eval_data import load_eval_data       # your loader
        jd_texts, resumes, labels = load_eval_data()
    except ImportError:
        raise SystemExit(
            "Couldn't import evaluation.eval_data.load_eval_data. "
            "Point this script at whatever module already loads your 60-resume / "
            "300-label eval set (the same data evaluate.py consumes)."
        )

    curve = run_active_learning_eval(
        jd_texts, resumes, labels,
        rounds=args.rounds,
        queries_per_round=args.k,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
    )
    print("\nFinal learning curve:")
    print(curve.to_string(index=False))
    if args.plot:
        plot_curve(curve)