"""scripts/sample_for_labeling.py on invented posts."""
import csv
import os

import pandas as pd
import pytest

from scripts.sample_for_labeling import COLUMNS, choose, main, write_csv
from src.common import config

T0_MS = 1_800_000_000_000


def posts(n=100):
    """One invented post every 6 seconds: 10 minutes."""
    return pd.DataFrame(
        {
            "post_id": [f"{i:064x}" for i in range(n)],
            "text": [f"Invented post {i}" for i in range(n)],
            "received_at_ms": [T0_MS + i * 6000 for i in range(n)],
        }
    )


def test_sample_is_after_the_warmup_without_repeats_and_repeatable():
    frame = posts()
    sample = choose(frame, n=20, seed=42, warmup_minutes=5)
    assert len(sample) == 20 and sample["post_id"].is_unique
    assert (sample["received_at_ms"] >= T0_MS + 5 * 60_000).all()
    # the same draw whatever order MongoDB returns the posts in
    again = choose(frame.sample(frac=1, random_state=1), n=20, seed=42, warmup_minutes=5)
    assert list(again["post_id"]) == list(sample["post_id"])
    assert list(choose(frame, n=20, seed=7, warmup_minutes=5)["post_id"]) != list(sample["post_id"])


def test_too_few_posts_after_the_warmup():
    with pytest.raises(SystemExit):
        choose(posts(), n=60, seed=42, warmup_minutes=5)  # only 50 after 5 minutes


def test_csv_has_three_columns_masked_handles_and_empty_labels(tmp_path):
    sample = pd.DataFrame({"post_id": ["abc"], "text": ["@someone.example.test thanks @united"], "received_at_ms": [0]})
    path = tmp_path / "out" / "sample.csv"
    write_csv(sample, str(path), keep_handles={"@united"})
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == COLUMNS
    assert rows[1] == ["abc", "@user thanks @united", ""]


def test_refuses_to_write_inside_the_repository(tmp_path):
    inside = os.path.join(config.BASE_DIR, "reports", "sample.csv")
    with pytest.raises(SystemExit):
        write_csv(posts(1), inside, keep_handles=set())
    with pytest.raises(SystemExit):
        main(["--output", inside])
    assert not os.path.exists(inside)
