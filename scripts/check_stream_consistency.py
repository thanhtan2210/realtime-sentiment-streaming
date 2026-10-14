"""Check that the stream's predictions are the offline model's predictions.

Usage (from the repo root):
    python -m scripts.check_stream_consistency

Reads the recorded run in reports/stream_snapshot/predictions.parquet, scores
the same tweets again with models/sentiment_pipeline.joblib, joins the true
labels of the test split and writes reports/stream_consistency.json:

  how many tweets got the same label from the stream and from the offline model,
  accuracy and macro-F1 of the stream's predictions against the true labels,
  and how many tweets the stream put in each class.

The output depends only on committed files, so rerunning it reproduces it.
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from src.common import config
from src.ml.data import LABELS, load_config, load_raw, prepare
from src.ml.train import macro_f1

SNAPSHOT_DIR = os.path.join(config.BASE_DIR, "reports", "stream_snapshot")
OUTPUT_PATH = os.path.join(config.BASE_DIR, "reports", "stream_consistency.json")


def consistency(stream: pd.DataFrame, model, test: pd.DataFrame) -> dict:
    """Compare the stream's `sentiment` per tweet with `model` and with the test labels.

    `stream` needs tweet_id, text and sentiment; `test` needs tweet_id and label.
    """
    labels = test.set_index("tweet_id")["label"]
    unknown = sorted(set(stream["tweet_id"]) - set(labels.index))
    if unknown or not stream["tweet_id"].is_unique:
        raise ValueError(f"the snapshot must hold each test tweet at most once; unknown ids: {unknown[:5]}")

    offline = model.predict(stream["text"].tolist())
    truth = labels.loc[stream["tweet_id"]].to_numpy()
    assigned = stream["sentiment"].value_counts()
    return {
        "tweets": int(len(stream)),
        "test_tweets": int(len(test)),
        "same_prediction_as_offline_model": int((stream["sentiment"].to_numpy() == offline).sum()),
        "stream_accuracy": round(float(accuracy_score(truth, stream["sentiment"])), 4),
        "stream_macro_f1": round(macro_f1(truth, stream["sentiment"]), 4),
        "stream_label_counts": {label: int(assigned.get(label, 0)) for label in LABELS},
    }


def run(stream: pd.DataFrame, model, test: pd.DataFrame, source: dict, output_path: str) -> dict:
    result = {"source": source, **consistency(stream, model, test)}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def main() -> int:
    stream = pd.read_parquet(os.path.join(SNAPSHOT_DIR, "predictions.parquet"))
    with open(os.path.join(SNAPSHOT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    _, test, _ = prepare(load_raw(), load_config())
    source = {
        "stream_predictions": "reports/stream_snapshot/predictions.parquet",
        "snapshot_commit": meta["commit"],
        "offline_model": "models/sentiment_pipeline.joblib",
        "labels": "test split of data/raw/Tweets.csv (src.ml.data)",
    }
    result = run(stream, joblib.load(config.MODEL_PATH), test, source, OUTPUT_PATH)
    print(
        f"{result['same_prediction_as_offline_model']} of {result['tweets']} stream predictions match the "
        f"offline model; macro-F1 {result['stream_macro_f1']}, accuracy {result['stream_accuracy']}"
    )
    print(f"Wrote {OUTPUT_PATH}")
    return 0 if result["same_prediction_as_offline_model"] == result["tweets"] == result["test_tweets"] else 1


if __name__ == "__main__":
    sys.exit(main())
