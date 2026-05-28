"""Privacy-safe recruiter-feedback store.

Stores feature vectors plus labels — never raw resume text — which satisfies the
project spec's constraint that "no sensitive personal data should be stored
beyond processing scope". `resume_id` is treated as an opaque identifier; if
your IDs contain candidate names, pass already-anonymised IDs from upstream.

Format: append-only JSONL at config.FEEDBACK_PATH. One feedback event per line.
This keeps the loop crash-safe (no rewriting) and easy to inspect / replay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

import config

# Map recruiter actions to binary labels for the classification layer.
_ACTION_TO_LABEL = {
    "shortlist": 1,
    "reject":    0,
}

@dataclass
class FeedbackRecord:
    """One recruiter judgement, stored as features (not raw text)."""
    jd_id: str
    resume_id: str
    semantic_score: float
    skill_score: float
    num_matched: int
    num_missing: int
    cross_lingual: int                   # 0/1 — int so it survives CSV roundtrips
    matched_skills: list = field(default_factory=list)
    action: str = "shortlist"
    label: int = 1
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.action in _ACTION_TO_LABEL:
            self.label = _ACTION_TO_LABEL[self.action]


def log_feedback(record: FeedbackRecord, path: str | None = None) -> None:
    """Append one feedback record to the JSONL store."""
    p = Path(path or config.FEEDBACK_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_feedback(path: str | None = None) -> pd.DataFrame:
    """Load all feedback as a DataFrame. Returns empty frame if no log yet."""
    p = Path(path or config.FEEDBACK_PATH)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def count_feedback(path: str | None = None) -> int:
    """Cheap count without loading every row into memory."""
    p = Path(path or config.FEEDBACK_PATH)
    if not p.exists():
        return 0
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def class_balance(path: str | None = None) -> tuple[int, int]:
    """(num_positives, num_negatives) — used to gate first training."""
    df = load_feedback(path)
    if df.empty:
        return 0, 0
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return pos, neg


def log_batch(records: Iterable[FeedbackRecord], path: str | None = None) -> int:
    """Append many records (used by the simulated-oracle harness)."""
    n = 0
    for r in records:
        log_feedback(r, path=path)
        n += 1
    return n


def clear_feedback(path: str | None = None) -> None:
    """Wipe the feedback log. Useful for re-running the simulated-oracle eval."""
    p = Path(path or config.FEEDBACK_PATH)
    if p.exists():
        p.unlink()