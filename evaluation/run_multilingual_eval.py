# evaluation/run_multilingual_eval.py
"""Run the multilingual evaluation.

What it does:
  1. Cross-lingual parity check — ranks every resume for each English JD and
     shows how the ENGLISH vs TRANSLATED copy of the correct resume score/rank.
  2. Threshold re-tuning — sweeps MATCH_THRESHOLD and recommends the F1-best value.
  3. Model A/B — headline metrics with the multilingual model vs the old
     monolingual model, to show the multilingual model is what makes
     cross-lingual matching work.

Run from the repository root:
    python -m evaluation.run_multilingual_eval
(The first run downloads the multilingual SBERT model, ~470 MB, once.)
"""
import config
import src.embeddings as embeddings
from src.matching import ResumeMatcher
from src.language import detect_language, language_name
from evaluation.evaluate import evaluate
from evaluation.multilingual_eval_data import load_dataset


def _reset_model(model_name: str):
    """Point the pipeline at a model and force a reload on next embed()."""
    config.MODEL_NAME = model_name
    embeddings._model = None


def cross_lingual_diagnostic(jd_texts, resumes, labels):
    print("\n" + "=" * 78)
    print("1. CROSS-LINGUAL PARITY CHECK  (multilingual model)")
    print("=" * 78)
    matcher = ResumeMatcher()
    rel = {(r.jd_id, r.resume_id): r.relevant for r in labels.itertuples()}
    for jd_id, jd_text in jd_texts.items():
        ranked = matcher.rank(jd_text, resumes)
        print(f"\nJD: {jd_id}   (language: {language_name(detect_language(jd_text))})")
        print(f"  {'rank':<5}{'resume':<14}{'lang':<10}{'final':<9}{'semantic':<10}{'relevant'}")
        for i, r in enumerate(ranked, 1):
            truth = rel.get((jd_id, r.resume_id), 0)
            mark = "  <-- relevant" if truth else ""
            print(f"  {i:<5}{r.resume_id:<14}{language_name(r.language):<10}"
                  f"{r.final_score:<9}{r.semantic_score:<10}{truth}{mark}")
        # parity: compare EN vs translated copy of the relevant resume
        rel_ids = [rid for (j, rid), t in rel.items() if j == jd_id and t]
        scores = {r.resume_id: r.final_score for r in ranked}
        en = [rid for rid in rel_ids if rid.endswith("_en")]
        tr = [rid for rid in rel_ids if not rid.endswith("_en")]
        if en and tr:
            gap = abs(scores[en[0]] - scores[tr[0]])
            print(f"  -> parity gap (EN vs translated relevant resume): {gap:.4f}")


def threshold_sweep(jd_texts, resumes, labels, lo=0.30, hi=0.70, step=0.02):
    print("\n" + "=" * 78)
    print("2. THRESHOLD RE-TUNING  (multilingual model)")
    print("=" * 78)
    matcher = ResumeMatcher()
    rel = {(r.jd_id, r.resume_id): r.relevant for r in labels.itertuples()}

    # Rank once per JD; cache (final_score, truth) so the sweep is cheap.
    pairs = []
    for jd_id, jd_text in jd_texts.items():
        for r in matcher.rank(jd_text, resumes):
            pairs.append((r.final_score, rel.get((jd_id, r.resume_id), 0)))

    def metrics_at(thr):
        tp = sum(1 for s, t in pairs if s >= thr and t == 1)
        fp = sum(1 for s, t in pairs if s >= thr and t == 0)
        fn = sum(1 for s, t in pairs if s < thr and t == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    print(f"  {'threshold':<11}{'precision':<11}{'recall':<9}{'f1'}")
    best = (-1.0, None)
    thr = lo
    while thr <= hi + 1e-9:
        p, r, f1 = metrics_at(round(thr, 2))
        flag = ""
        if f1 > best[0]:
            best = (f1, round(thr, 2))
        print(f"  {round(thr,2):<11}{round(p,3):<11}{round(r,3):<9}{round(f1,3)}{flag}")
        thr += step
    print(f"\n  -> recommended MATCH_THRESHOLD = {best[1]}  (F1 = {round(best[0],3)})")
    print(f"     current config value         = {config.MATCH_THRESHOLD}")


def model_ab(jd_texts, resumes, labels):
    print("\n" + "=" * 78)
    print("3. MODEL A/B  (does the multilingual model actually help?)")
    print("=" * 78)
    original = config.MODEL_NAME
    try:
        for label, model in [("multilingual", config.MODEL_NAME),
                             ("monolingual ", config.MONOLINGUAL_MODEL_NAME)]:
            _reset_model(model)
            m = evaluate(jd_texts, resumes, labels)
            print(f"\n  [{label}]  model = {model}")
            print(f"     precision={m['precision']}  recall={m['recall']}  f1={m['f1']}  "
                  f"mrr={m['mrr']}  ndcg={m['ndcg']}")
            print(f"     cosine matched={m['cosine_matched_mean']}  "
                  f"unmatched={m['cosine_unmatched_mean']}  "
                  f"(separation={round(m['cosine_matched_mean']-m['cosine_unmatched_mean'],4)})")
    finally:
        _reset_model(original)
    print("\n  Expect the multilingual model to show a larger matched/unmatched")
    print("  cosine separation — it aligns the translated resumes with the English JDs.")


def main():
    jd_texts, resumes, labels = load_dataset()
    print(f"Dataset: {len(jd_texts)} JDs, {len(resumes)} resumes "
          f"({sum(1 for k in resumes if not k.endswith('_en'))} translated), "
          f"{len(labels)} labels.")
    cross_lingual_diagnostic(jd_texts, resumes, labels)
    threshold_sweep(jd_texts, resumes, labels)
    model_ab(jd_texts, resumes, labels)


if __name__ == "__main__":
    main()