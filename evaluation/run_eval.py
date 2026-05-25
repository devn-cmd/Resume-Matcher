"""
Loads the prepared evaluation files and runs the harness.

Run from the project root:
    python -m evaluation.run_eval
"""
import pandas as pd
from evaluation.evaluate import evaluate


def _load_dict(path, key_col, val_col):
    df = pd.read_csv(path)
    return dict(zip(df[key_col].astype(str), df[val_col].astype(str)))


def main():
    resumes = _load_dict("data/eval/resumes.csv", "resume_id", "text")
    jd_texts = _load_dict("data/eval/jds.csv", "jd_id", "text")
    labels = pd.read_csv("data/eval/labels.csv")

    print(f"Evaluating {len(resumes)} resumes against {len(jd_texts)} job descriptions...\n")
    metrics = evaluate(jd_texts, resumes, labels)

    print("=== Evaluation results ===")
    for name, value in metrics.items():
        print(f"{name:26s} {value}")


if __name__ == "__main__":
    main()