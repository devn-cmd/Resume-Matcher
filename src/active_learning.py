"""Active-learning layer for the resume matcher.

This module implements the two halves of the active-learning loop:

* **Query strategy** (`select_review_queue`): uncertainty sampling — surface the
  candidates whose blended (or adaptive) score sits closest to the suitability
  threshold, so each recruiter label moves the decision boundary maximally.

* **Learning component** (`ActiveLearningRanker`): an L2-regularised
  logistic-regression "classification layer" (per the spec's modeling
  requirement) over a small, interpretable feature set. The coefficients are
  directly inspectable — i.e. you can read off which features drive each
  prediction — which satisfies the spec's interpretability requirement without
  the opacity of tree ensembles.

Design choices:

* The model is *additive* — until enough feedback is accumulated to train
  responsibly (config.MIN_FEEDBACK_TO_TRAIN, with both classes represented),
  scoring falls back to the existing static blend. Cold-start safe.
* Regularisation is set tight (small C) on purpose: the model will frequently
  be trained on tens-to-hundreds of feedback points, where overfitting is the
  realistic failure mode.
* `recruiter_preference_score` — a Laplace-smoothed "lift" of each canonical
  skill under shortlist vs reject — is a learned signal carried as a feature.
  This captures *which skills* recruiters keep favouring (Kubernetes, AWS,
  Docker...) without hard-coding any rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from src.feedback import load_feedback, class_balance


# The feature vector consumed by the LR re-ranker. Keep this list small and
# named so the coefficients remain readable.
FEATURE_NAMES: list[str] = [
    "semantic_score",
    "skill_score",
    "num_matched",
    "num_missing",
    "cross_lingual",
    "preference_score",
]


# ---------------------------------------------------------------------------
# Recruiter preference weights — learned per-skill lift from feedback
# ---------------------------------------------------------------------------

def compute_skill_preferences(feedback_df: pd.DataFrame, alpha: float = 1.0) -> dict[str, float]:
    """Laplace-smoothed lift for each canonical skill seen in feedback.

    Returns a dict {skill: weight in [-0.5, 0.5]}, where positive means the
    skill is associated with shortlists, negative with rejects, and 0 is
    neutral. Smoothing protects against skills that appear once.
    """
    if feedback_df.empty or "matched_skills" not in feedback_df.columns:
        return {}
    counts: dict[str, list[int]] = {}     # skill -> [pos, total]
    for _, row in feedback_df.iterrows():
        label = int(row["label"])
        for skill in (row.get("matched_skills") or []):
            s = str(skill).lower()
            stat = counts.setdefault(s, [0, 0])
            stat[0] += label
            stat[1] += 1
    weights: dict[str, float] = {}
    for s, (pos, total) in counts.items():
        # smoothed P(shortlist | skill present) - 0.5
        weights[s] = (pos + alpha) / (total + 2 * alpha) - 0.5
    return weights


def preference_score(matched_skills: Iterable[str], weights: dict[str, float]) -> float:
    """Aggregate skill-level preferences into a single feature in [0, 1].

    Neutral (0.5) when no learned weights overlap the candidate's matched
    skills — important for cold-start so the feature doesn't actively penalise
    candidates the recruiter has never expressed a view about.
    """
    if not weights or not matched_skills:
        return 0.5
    vals = [weights[s.lower()] for s in matched_skills if s.lower() in weights]
    if not vals:
        return 0.5
    return float(np.clip(np.mean(vals) + 0.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# The classification layer
# ---------------------------------------------------------------------------

@dataclass
class TrainingReport:
    n_samples: int
    n_positive: int
    n_negative: int
    trained: bool
    reason: str = ""              # populated when trained=False
    coefficients: dict | None = None


class ActiveLearningRanker:
    """An interpretable, sklearn-backed classification layer.

    Public surface:
      * fit(feedback_df)   — returns TrainingReport
      * is_trained()       — bool
      * predict_proba(feats) — float in [0, 1], P(suitable)
      * coefficients()     — dict of feature -> learned weight (interpretability)
      * save(path) / load(path)
    """

    def __init__(self):
        self.model: LogisticRegression | None = None
        self.skill_weights: dict[str, float] = {}
        self.n_train_samples: int = 0

    # ----- training -----------------------------------------------------------

    def fit(self, feedback_df: pd.DataFrame) -> TrainingReport:
        n = len(feedback_df)
        pos = int((feedback_df["label"] == 1).sum()) if n else 0
        neg = int((feedback_df["label"] == 0).sum()) if n else 0

        if n < config.MIN_FEEDBACK_TO_TRAIN:
            return TrainingReport(n, pos, neg, trained=False,
                                  reason=f"need >= {config.MIN_FEEDBACK_TO_TRAIN} samples (have {n})")
        if pos < config.MIN_PER_CLASS_TO_TRAIN or neg < config.MIN_PER_CLASS_TO_TRAIN:
            return TrainingReport(n, pos, neg, trained=False,
                                  reason="need both shortlist and reject examples")

        # Learn skill-level preferences from the same data, then derive the
        # preference_score column for the training matrix.
        self.skill_weights = compute_skill_preferences(feedback_df)

        df = feedback_df.copy()
        df["preference_score"] = df["matched_skills"].apply(
            lambda sk: preference_score(sk or [], self.skill_weights)
        )
        X = df[FEATURE_NAMES].astype(float).to_numpy()
        y = df["label"].astype(int).to_numpy()

        self.model = LogisticRegression(
            C=config.LR_REGULARIZATION_C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=config.LR_RANDOM_SEED,
        )
        self.model.fit(X, y)
        self.n_train_samples = n
        return TrainingReport(n, pos, neg, trained=True, coefficients=self.coefficients())

    def is_trained(self) -> bool:
        return self.model is not None

    # ----- inference ----------------------------------------------------------

    def predict_proba(self, features: dict) -> float | None:
        """Return P(suitable) for one feature dict, or None if untrained."""
        if not self.is_trained():
            return None
        x = np.array([[float(features[name]) for name in FEATURE_NAMES]])
        return float(self.model.predict_proba(x)[0, 1])

    def coefficients(self) -> dict[str, float]:
        if not self.is_trained():
            return {}
        return {n: float(c) for n, c in zip(FEATURE_NAMES, self.model.coef_[0])}

    # ----- persistence --------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        p = Path(path or config.ACTIVE_MODEL_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "skill_weights": self.skill_weights,
            "n_train_samples": self.n_train_samples,
        }, p)

    def load(self, path: str | None = None) -> bool:
        p = Path(path or config.ACTIVE_MODEL_PATH)
        if not p.exists():
            return False
        bundle = joblib.load(p)
        self.model = bundle.get("model")
        self.skill_weights = bundle.get("skill_weights", {})
        self.n_train_samples = bundle.get("n_train_samples", 0)
        return self.is_trained()


# ---------------------------------------------------------------------------
# Convenience: train-from-disk + cache
# ---------------------------------------------------------------------------

def load_or_train_from_disk() -> ActiveLearningRanker:
    """Best-effort: load a saved model; if absent, try to train from feedback."""
    ranker = ActiveLearningRanker()
    if ranker.load():
        return ranker
    df = load_feedback()
    if not df.empty:
        ranker.fit(df)
        if ranker.is_trained():
            ranker.save()
    return ranker


# ---------------------------------------------------------------------------
# Query strategy: uncertainty sampling
# ---------------------------------------------------------------------------

def _decision_score(result, ranker: ActiveLearningRanker | None) -> float:
    """The score used to measure distance from the decision boundary."""
    if ranker is not None and ranker.is_trained() and getattr(result, "adaptive_score", None) is not None:
        return result.adaptive_score
    return result.final_score


def _threshold(ranker: ActiveLearningRanker | None) -> float:
    return config.ADAPTIVE_THRESHOLD if (ranker and ranker.is_trained()) else config.MATCH_THRESHOLD


def uncertainty(result, ranker: ActiveLearningRanker | None = None) -> float:
    """0 = most certain, 1 = sitting exactly on the boundary."""
    score = _decision_score(result, ranker)
    return 1.0 - min(1.0, 2.0 * abs(score - _threshold(ranker)))


def select_review_queue(results, ranker: ActiveLearningRanker | None = None,
                        k: int | None = None) -> list:
    """Top-k candidates the model is least sure about (most useful to label next).

    Tiebreak: cross-lingual candidates first (documented weak spot), then the
    lower static score (slight preference for the harder side of the boundary).
    """
    k = k or config.REVIEW_QUEUE_SIZE
    scored = [
        (r, uncertainty(r, ranker), int(getattr(r, "cross_lingual", False)), r.final_score)
        for r in results
    ]
    # higher uncertainty first; cross-lingual first on tie; lower final_score first
    scored.sort(key=lambda t: (-t[1], -t[2], t[3]))
    queue = [r for r, _, _, _ in scored[:k]]
    # set the convenience flag for UI rendering
    in_queue = set(id(r) for r in queue)
    for r in results:
        if hasattr(r, "is_uncertain"):
            r.is_uncertain = id(r) in in_queue
    return queue


# ---------------------------------------------------------------------------
# Apply the trained re-ranker to a static-ranked list
# ---------------------------------------------------------------------------

def features_from_result(result, skill_weights: dict[str, float]) -> dict:
    """Build the feature dict the ranker expects from a MatchResult."""
    return {
        "semantic_score":   float(result.semantic_score),
        "skill_score":      float(result.skill_score),
        "num_matched":      float(len(result.matched_skills)),
        "num_missing":      float(len(result.missing_skills)),
        "cross_lingual":    1.0 if getattr(result, "cross_lingual", False) else 0.0,
        "preference_score": preference_score(result.matched_skills, skill_weights),
    }


def apply_active_learning(results, ranker: ActiveLearningRanker | None) -> list:
    """Re-score and re-sort a list of MatchResults with the trained re-ranker.

    No-op when the ranker is missing or untrained — the caller gets back the
    same list ordered by the original static score, which is exactly what the
    pipeline did before this feature existed.
    """
    if ranker is None or not ranker.is_trained():
        return results
    for r in results:
        feats = features_from_result(r, ranker.skill_weights)
        r.recruiter_preference_score = feats["preference_score"]
        r.adaptive_score = ranker.predict_proba(feats)
    results.sort(
        key=lambda r: (r.adaptive_score if r.adaptive_score is not None else r.final_score),
        reverse=True,
    )
    return results