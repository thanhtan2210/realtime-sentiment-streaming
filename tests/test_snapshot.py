"""The committed stream snapshot is complete and agrees with itself and with the test split."""
import json
import os

import pandas as pd
import pytest

from src.common import config
from src.ml.data import LABELS, load_config, load_raw
from src.stream.producer import select_test_tweets

SNAPSHOT_DIR = os.path.join(config.BASE_DIR, "reports", "stream_snapshot")
MAX_BYTES = 5 * 1024 * 1024


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(os.path.join(SNAPSHOT_DIR, "predictions.parquet"))


@pytest.fixture(scope="module")
def windows():
    return pd.read_parquet(os.path.join(SNAPSHOT_DIR, "windows.parquet"))


@pytest.fixture(scope="module")
def meta():
    with open(os.path.join(SNAPSHOT_DIR, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def test_snapshot_is_small_and_carries_its_license():
    names = sorted(os.listdir(SNAPSHOT_DIR))
    assert names == ["README.md", "meta.json", "predictions.parquet", "windows.parquet"]
    assert sum(os.path.getsize(os.path.join(SNAPSHOT_DIR, name)) for name in names) < MAX_BYTES
    with open(os.path.join(SNAPSHOT_DIR, "README.md"), encoding="utf-8") as f:
        assert "CC BY-NC-SA 4.0" in f.read()
    with open(os.path.join(config.BASE_DIR, "data", "raw", "SOURCE.md"), encoding="utf-8") as f:
        assert "reports/stream_snapshot/" in f.read()


def test_predictions_are_exactly_the_test_tweets_without_labels(predictions):
    test = select_test_tweets(load_raw(), load_config())
    assert list(predictions.columns) == [
        "tweet_id",
        "airline",
        "text",
        "created_at",
        "sentiment",
        "produced_at_ms",
        "written_at",
    ]
    assert predictions["tweet_id"].is_unique
    assert sorted(predictions["tweet_id"]) == sorted(test["tweet_id"])
    assert set(predictions["sentiment"]) <= set(LABELS)
    assert str(predictions["created_at"].dt.tz) == "UTC"
    by_id = predictions.set_index("tweet_id")["created_at"]
    assert (by_id.loc[test["tweet_id"]].to_numpy() == test["created_at"].to_numpy()).all()


def test_window_totals_add_up_to_the_number_of_predictions(predictions, windows, meta):
    assert int(windows["total"].sum()) == len(predictions) == meta["window_total"]
    assert (windows[LABELS].sum(axis=1) == windows["total"]).all()
    assert meta["records"] == {"predictions": len(predictions), "airline_windows": len(windows)}
    assert meta["tweets_sent"] == len(predictions)
    assert meta["rows_dropped_by_watermark"] == 0
    assert windows["_id"].is_unique


def test_windows_agree_with_the_predictions_row_by_row(predictions, windows):
    """Recount every airline and hour from the predictions; Spark's stored counts must match."""
    hour = predictions["created_at"].dt.floor(pd.Timedelta(load_config()["stream"]["window"]))
    recount = (
        predictions.groupby(["airline", hour, "sentiment"]).size().unstack("sentiment", fill_value=0)
    )
    stored = windows.set_index(["airline", "window_start"])[LABELS]
    assert len(recount) == len(stored)
    assert (recount.reindex(stored.index)[LABELS].to_numpy() == stored.to_numpy()).all()
