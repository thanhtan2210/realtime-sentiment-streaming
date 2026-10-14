"""scripts/check_stream_consistency.py on the small fixture, and the committed report."""
import json
import os

import pytest

from scripts.check_stream_consistency import OUTPUT_PATH, consistency, run
from src.ml.data import LABELS, prepare
from src.ml.train import build_pipeline, macro_f1
from tests.test_train import runtime_keys


@pytest.fixture
def fitted(raw, cfg):
    """A model trained on the fixture's training tweets, and its test tweets as a 'stream'."""
    train, test, _ = prepare(raw, cfg)
    model = build_pipeline().fit(train["text"], train["label"])
    stream = test[["tweet_id", "text"]].assign(sentiment=model.predict(test["text"]))
    return model, test, stream


def test_identical_predictions_are_counted_as_matching(fitted):
    model, test, stream = fitted
    result = consistency(stream, model, test)
    assert result["tweets"] == result["test_tweets"] == len(test)
    assert result["same_prediction_as_offline_model"] == len(test)
    assert result["stream_macro_f1"] == round(macro_f1(test["label"], stream["sentiment"]), 4)
    assert sum(result["stream_label_counts"].values()) == len(test)
    assert list(result["stream_label_counts"]) == LABELS


def test_a_different_stream_prediction_is_detected(fitted):
    model, test, stream = fitted
    changed = stream.copy()
    first = changed.index[0]
    changed.loc[first, "sentiment"] = next(l for l in LABELS if l != changed.loc[first, "sentiment"])
    result = consistency(changed, model, test)
    assert result["same_prediction_as_offline_model"] == len(test) - 1
    # the order of the stream's rows does not matter
    assert consistency(changed.iloc[::-1], model, test) == result


def test_tweets_outside_the_test_split_are_rejected(fitted):
    model, test, stream = fitted
    stranger = stream.head(1).assign(tweet_id="not-a-test-tweet")
    with pytest.raises(ValueError):
        consistency(stranger, model, test)


def test_report_has_no_runtime_fields_and_is_reproducible(fitted, tmp_path):
    model, test, stream = fitted
    source = {"stream_predictions": "fixture"}
    first = run(stream, model, test, source, str(tmp_path / "a" / "report.json"))
    run(stream, model, test, source, str(tmp_path / "b" / "report.json"))
    assert runtime_keys(first) == []
    assert (tmp_path / "a" / "report.json").read_bytes() == (tmp_path / "b" / "report.json").read_bytes()


def test_committed_report_says_the_stream_matches_the_model():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    assert runtime_keys(report) == []
    assert report["tweets"] == report["test_tweets"] == report["same_prediction_as_offline_model"]
    assert sum(report["stream_label_counts"].values()) == report["tweets"]
    with open(os.path.join(os.path.dirname(OUTPUT_PATH), "stream_snapshot", "meta.json"), encoding="utf-8") as f:
        assert report["source"]["snapshot_commit"] == json.load(f)["commit"]
