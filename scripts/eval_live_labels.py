"""How well does the 2015 airline model do on live Bluesky posts? Scores it on hand labels.

Usage (from the repo root):
    python -m scripts.eval_live_labels --labels PATH

PATH is the CSV from scripts/sample_for_labeling.py with the `label` column
filled in (negative, neutral, positive or skip). It holds post text, so it must
be outside the repository. Rows labelled `skip` and rows left empty are not
scored. The current model (models/sentiment_pipeline.joblib) predicts each
`text`; the scores are macro-F1, F1 per class, accuracy and the confusion
matrix, with paired bootstrap 95% CIs (1000 resamples, seed 42), next to a
baseline that always predicts the most common hand label and next to the
model's test score on the 2015 tweets (models/metrics.json).

Writes reports/live_domain_shift.json with aggregates only: no text and no
sample ids.
"""
import argparse
import hashlib
import json
import os

import joblib
import pandas as pd

from scripts.benchmark_report import commit_sha, utc_now
from src.common import config
from src.ml.data import LABELS
from src.ml.train import classification_metrics, paired_bootstrap

OUTPUT_PATH = os.path.join(config.BASE_DIR, "reports", "live_domain_shift.json")
SKIP = "skip"
N_RESAMPLES, SEED = 1000, 42
NOTE = (
    "Single annotator. Posts are a random sample of English Bluesky posts from one live run, "
    "labelled by hand without seeing the model's label."
)


def inside_repo(path: str) -> bool:
    root = os.path.abspath(config.BASE_DIR)
    return os.path.abspath(path).startswith(root + os.sep)


def read_labels(path: str) -> tuple[pd.DataFrame, dict]:
    """The scored rows (text, label) and the counts of rows read, skipped and left empty."""
    if inside_repo(path):
        raise SystemExit(f"Refusing to read a labels file inside the repository: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} has no column {', '.join(sorted(missing))}")
    labels = frame["label"].str.strip().str.lower()
    unknown = sorted(set(labels) - set(LABELS) - {SKIP, ""})
    if unknown:
        raise SystemExit(f"Unknown labels in {path}: {unknown}; use {', '.join(LABELS)}, {SKIP} or leave empty")
    counts = {"rows": int(len(frame)), "skip": int((labels == SKIP).sum()), "empty": int((labels == "").sum())}
    scored = pd.DataFrame({"text": frame["text"], "label": labels})[labels.isin(LABELS)].reset_index(drop=True)
    counts["scored"] = int(len(scored))
    if scored.empty:
        raise SystemExit(f"No row of {path} has a label to score")
    return scored, counts


def evaluate(scored: pd.DataFrame, predict, reference: dict) -> dict:
    """Scores of `predict` (texts -> labels) and of the majority baseline on the hand labels."""
    predicted = list(predict(scored["text"].tolist()))
    majority = scored["label"].value_counts().sort_index().idxmax()  # ties: the first label alphabetically
    baseline = [majority] * len(scored)
    model = classification_metrics(scored["label"], predicted)
    base = classification_metrics(scored["label"], baseline)
    ci = paired_bootstrap(scored["label"], predicted, baseline, N_RESAMPLES, SEED)
    model["macro_f1_ci95"], base["macro_f1_ci95"] = ci["model"], ci["baseline"]
    reference_ci = reference.get("macro_f1_ci95")
    return {
        "hand_label_counts": {label: int((scored["label"] == label).sum()) for label in LABELS},
        "model_label_counts": {label: int(sum(p == label for p in predicted)) for label in LABELS},
        "model": model,
        "baseline": {"predicts": majority, **base},
        "macro_f1_difference_vs_baseline": {
            "value": round(model["macro_f1"] - base["macro_f1"], 4),
            "ci95": ci["difference"],
        },
        "vs_2015_test_set": {
            "source": "models/metrics.json (test.model)",
            "macro_f1_2015": reference.get("macro_f1"),
            "macro_f1_ci95_2015": reference_ci,
            "macro_f1_live_minus_2015": round(model["macro_f1"] - reference["macro_f1"], 4),
            "cis_overlap": None
            if reference_ci is None
            else bool(model["macro_f1_ci95"][1] >= reference_ci[0] and reference_ci[1] >= model["macro_f1_ci95"][0]),
        },
        "bootstrap": {"method": "paired percentile", "n_resamples": N_RESAMPLES, "seed": SEED},
    }


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, help="labelled CSV, outside the repository")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--metrics", default=config.METRICS_PATH)
    args = parser.parse_args(argv)

    scored, counts = read_labels(args.labels)
    with open(args.metrics, encoding="utf-8") as f:
        reference = json.load(f)["test"]["model"]
    model = joblib.load(args.model)
    result = evaluate(scored, model.predict, reference)
    report = {
        "generated_at": utc_now(),
        "commit": commit_sha(),
        "note": NOTE,
        "labels_file_sha256": sha256_of(args.labels),
        "model_file": os.path.relpath(os.path.abspath(args.model), config.BASE_DIR).replace(os.sep, "/"),
        "rows": counts,
        "n": counts["scored"],
        **result,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(
        f"Wrote {args.output}: n={report['n']} (skip {counts['skip']}, empty {counts['empty']}), "
        f"macro-F1 {result['model']['macro_f1']} vs {reference['macro_f1']} on the 2015 test set"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
