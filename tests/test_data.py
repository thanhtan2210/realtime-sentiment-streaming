import pandas as pd
import pytest

from src.ml.data import COLUMNS, LABELS, clean, prepare, time_split
from tests.conftest import TEST_START


def test_clean_keeps_only_modelling_columns_and_three_labels(raw):
    df, stats = clean(raw)
    assert list(df.columns) == COLUMNS
    assert set(df["label"]) == set(LABELS)
    assert stats["raw_rows"] == len(raw)
    assert stats["rows_with_valid_label_and_text"] == len(raw) - 1  # the "mixed" row


def test_created_at_is_timezone_aware_and_sorted(raw):
    df, _ = clean(raw)
    assert df["created_at"].dt.tz is not None
    assert df["created_at"].is_monotonic_increasing
    # 2015-02-17 08:00 -0800 is 16:00 UTC
    assert df["created_at"].iloc[0] == pd.Timestamp("2015-02-17T16:00:00Z")


def test_duplicate_texts_are_dropped_before_the_split(raw, cfg):
    train, test, stats = prepare(raw, cfg)
    assert stats["duplicate_text_rows_removed"] == 1
    assert stats["rows"] == len(train) + len(test)

    # The earlier copy (in the training period) is the one kept. Had the
    # split run first, the later copy would have survived in the test set.
    assert (train["text"] == "my bag is lost again").sum() == 1
    assert (test["text"] == "my bag is lost again").sum() == 0
    assert not set(train["text"]) & set(test["text"])


def test_no_test_tweet_is_before_the_cutoff(raw, cfg):
    train, test, _ = prepare(raw, cfg)
    cutoff = pd.Timestamp(TEST_START)
    assert len(train) > 0 and len(test) > 0
    assert (test["created_at"] >= cutoff).all()
    assert (train["created_at"] < cutoff).all()
    assert "last tweet before the cut" in set(train["text"])
    assert "first tweet after the cut" in set(test["text"])


def test_cutoff_without_utc_offset_is_rejected(raw):
    df, _ = clean(raw)
    with pytest.raises(ValueError):
        time_split(df, "2015-02-23")
