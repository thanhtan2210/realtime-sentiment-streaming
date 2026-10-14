"""Train the sentiment model and write models/metrics.json.

Usage (from the repo root):
    python -m src.ml.train

Steps: de-duplicate, split by time, pick hyper-parameters with time-ordered
cross-validation on the training set, refit on the whole training set, then
score the test set once against a majority-class baseline.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.common import config, logger as common_logger
from src.ml.data import LABELS, load_config, load_raw, prepare

logger = common_logger.get_logger(__name__)


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def majority_baseline(y_train: pd.Series) -> DummyClassifier:
    """Always predicts the most frequent label of the training set."""
    return DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)


def param_grid(cfg: dict) -> dict:
    grid = dict(cfg["cv"]["grid"])
    # YAML has no tuples; TfidfVectorizer needs ngram_range as a tuple.
    grid["tfidf__ngram_range"] = [tuple(r) for r in grid["tfidf__ngram_range"]]
    return grid


def select_model(train: pd.DataFrame, cfg: dict) -> GridSearchCV:
    """Grid search with TimeSeriesSplit; `train` must already be ordered by time."""
    search = GridSearchCV(
        build_pipeline(cfg["seed"]),
        param_grid(cfg),
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"]),
        refit=True,
    )
    return search.fit(train["text"], train["label"])


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def macro_f1_from_codes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same value as `macro_f1`, for labels coded as positions in LABELS.

    Plain numpy so the bootstrap loop stays fast.
    """
    k = len(LABELS)
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    support_plus_predicted = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(
        2 * np.diag(cm), support_plus_predicted, out=np.zeros(k), where=support_plus_predicted > 0
    )
    return float(f1.mean())


def to_codes(labels) -> np.ndarray:
    return pd.Categorical(labels, categories=LABELS).codes.astype(np.int64)


def classification_metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "f1_per_class": {label: round(float(f), 4) for label, f in zip(LABELS, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def ci95(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def paired_bootstrap(y_true, pred_model, pred_baseline, n_resamples: int, seed: int) -> dict:
    """95% CIs for macro-F1, using the same resampled rows for model and baseline."""
    y_true, pred_model, pred_baseline = map(to_codes, (y_true, pred_model, pred_baseline))
    rng = np.random.default_rng(seed)
    model_scores = np.empty(n_resamples)
    baseline_scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        model_scores[i] = macro_f1_from_codes(y_true[idx], pred_model[idx])
        baseline_scores[i] = macro_f1_from_codes(y_true[idx], pred_baseline[idx])
    return {
        "model": ci95(model_scores),
        "baseline": ci95(baseline_scores),
        "difference": ci95(model_scores - baseline_scores),
    }


def label_counts(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def split_summary(part: pd.DataFrame) -> dict:
    return {
        "rows": int(len(part)),
        "first_tweet": part["created_at"].min().isoformat(),
        "last_tweet": part["created_at"].max().isoformat(),
        "label_counts": label_counts(part["label"]),
    }


def cv_table(search: GridSearchCV) -> list[dict]:
    results = search.cv_results_
    fold_keys = [f"split{i}_test_score" for i in range(search.n_splits_)]
    return [
        {
            "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
            "mean_macro_f1": round(float(results["mean_test_score"][i]), 4),
            "std_macro_f1": round(float(results["std_test_score"][i]), 4),
            "fold_macro_f1": [round(float(results[k][i]), 4) for k in fold_keys],
            "rank": int(results["rank_test_score"][i]),
        }
        for i, params in enumerate(results["params"])
    ]


def by_airline(test: pd.DataFrame, pred_model, pred_baseline, n_resamples: int, seed: int) -> list:
    rows = []
    for airline, idx in sorted(test.groupby("airline").indices.items()):
        y_true = test["label"].to_numpy()[idx]
        ci = paired_bootstrap(y_true, pred_model[idx], pred_baseline[idx], n_resamples, seed)
        rows.append(
            {
                "airline": airline,
                "n": int(len(idx)),
                "label_counts": label_counts(pd.Series(y_true)),
                "macro_f1": round(macro_f1(y_true, pred_model[idx]), 4),
                "macro_f1_ci95": ci["model"],
            }
        )
    return rows


def run(raw: pd.DataFrame, cfg: dict, model_out: str, metrics_out: str) -> dict:
    seed = cfg["seed"]
    n_resamples = cfg["bootstrap"]["n_resamples"]

    train, test, data_stats = prepare(raw, cfg)
    logger.info(f"Rows after de-duplication: {data_stats['rows']} (train {len(train)}, test {len(test)})")

    baseline = majority_baseline(train["label"])
    search = select_model(train, cfg)
    model = search.best_estimator_  # refit on the whole training set
    logger.info(f"Best CV macro-F1 {search.best_score_:.4f} with {search.best_params_}")

    # The only place the test set is scored.
    pred_model = model.predict(test["text"])
    pred_baseline = baseline.predict(np.zeros((len(test), 1)))
    y_test = test["label"].to_numpy()

    cis = paired_bootstrap(y_test, pred_model, pred_baseline, n_resamples, seed)
    model_metrics = classification_metrics(y_test, pred_model)
    baseline_metrics = classification_metrics(y_test, pred_baseline)
    cv_results = cv_table(search)

    metrics = {
        "data": data_stats,
        "split": {
            "test_start": cfg["split"]["test_start"],
            "test_share": round(len(test) / (len(train) + len(test)), 4),
            "train": split_summary(train),
            "test": split_summary(test),
            "texts_in_both_train_and_test": int(test["text"].isin(set(train["text"])).sum()),
        },
        "cv": {
            "method": "TimeSeriesSplit",
            "n_splits": cfg["cv"]["n_splits"],
            "scoring": "macro_f1",
            "results": cv_results,
            "best_params": cv_results[int(search.best_index_)]["params"],
        },
        "test": {
            "labels": LABELS,
            "baseline_predicts": str(baseline.classes_[np.argmax(baseline.class_prior_)]),
            "baseline": {**baseline_metrics, "macro_f1_ci95": cis["baseline"]},
            "model": {**model_metrics, "macro_f1_ci95": cis["model"]},
            "macro_f1_difference": {
                "value": round(model_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4),
                "ci95": cis["difference"],
            },
            "bootstrap": {"method": "paired percentile", "n_resamples": n_resamples, "seed": seed},
            "by_airline": by_airline(test, pred_model, pred_baseline, n_resamples, seed),
        },
    }

    # CountVectorizer caches id(stop_words), a memory address that changes between
    # processes. Nothing reads it after fitting; drop it so the file is reproducible.
    vars(model.named_steps["tfidf"]).pop("_stop_words_id", None)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    with open(metrics_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    return metrics


def main():
    metrics = run(load_raw(), load_config(), config.MODEL_PATH, config.METRICS_PATH)
    test = metrics["test"]
    logger.info(f"Test macro-F1: model {test['model']['macro_f1']}, baseline {test['baseline']['macro_f1']}")
    logger.info(f"Wrote {config.MODEL_PATH} and {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
