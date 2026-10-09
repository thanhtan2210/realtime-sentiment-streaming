import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common import config
from src.ml.data import LABELS, prepare
from src.ml.train import (
    build_pipeline,
    macro_f1,
    macro_f1_from_codes,
    majority_baseline,
    run,
    to_codes,
)

RUNTIME_KEY_PARTS = ("time", "date", "duration", "elapsed", "seconds", "generated", "started", "finished")


def all_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from all_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from all_keys(item)


def runtime_keys(metrics: dict) -> list[str]:
    return [k for k in all_keys(metrics) if any(part in k.lower() for part in RUNTIME_KEY_PARTS)]


def test_baseline_predicts_the_training_majority_class():
    y_train = pd.Series(["neutral"] * 5 + ["negative"] * 3 + ["positive"] * 2)
    baseline = majority_baseline(y_train)
    assert set(baseline.predict(np.zeros((20, 1)))) == {"neutral"}


def test_pipeline_predicts_only_the_three_labels(raw, cfg):
    train, test, _ = prepare(raw, cfg)
    pipe = build_pipeline().fit(train["text"], train["label"])
    assert set(pipe.predict(test["text"])) <= set(LABELS)


def test_committed_model_predicts_only_the_three_labels():
    pipe = joblib.load(config.MODEL_PATH)
    texts = ["@united my flight was cancelled and nobody helped", "@delta thanks, great crew!", ""]
    assert set(pipe.predict(texts)) <= set(LABELS)
    assert list(pipe.classes_) == LABELS


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.choice(LABELS, size=200)
    y_pred = rng.choice(LABELS[:2], size=200)  # "positive" is never predicted
    assert macro_f1_from_codes(to_codes(y_true), to_codes(y_pred)) == pytest.approx(
        macro_f1(y_true, y_pred)
    )


def test_run_writes_metrics_without_runtime_keys(raw, cfg, tmp_path):
    model_out, metrics_out = str(tmp_path / "model.joblib"), str(tmp_path / "metrics.json")
    run(raw, cfg, model_out, metrics_out)
    with open(metrics_out, encoding="utf-8") as f:
        metrics = json.load(f)

    assert runtime_keys(metrics) == []
    assert metrics["split"]["texts_in_both_train_and_test"] == 0
    assert metrics["test"]["baseline_predicts"] == "negative"
    assert len(metrics["cv"]["results"]) == 4
    assert sum(row["n"] for row in metrics["test"]["by_airline"]) == metrics["split"]["test"]["rows"]
    assert os.path.exists(model_out)


def test_run_is_reproducible(raw, cfg, tmp_path):
    outputs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        model_out = tmp_path / name / "model.joblib"
        metrics_out = tmp_path / name / "metrics.json"
        run(raw, cfg, str(model_out), str(metrics_out))
        outputs.append((model_out.read_bytes(), metrics_out.read_bytes()))
    assert outputs[0] == outputs[1]


def test_committed_metrics_have_no_runtime_keys():
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    assert runtime_keys(metrics) == []
