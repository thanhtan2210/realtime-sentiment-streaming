"""Pick a random sample of live posts for manual labeling; writes a CSV outside the repository.

Usage (from the repo root, after a live run):
    python -m scripts.sample_for_labeling --n 200 --seed 42 --output PATH

Draws --n posts written after the warm-up (the first --warmup-minutes after
the first post was received), without replacement, with a fixed seed. The CSV
has three columns: `sample_id` (the post_id hash), `text` (other people's
@handles replaced with "@user") and `label`, left empty for the annotator
(negative, neutral, positive or skip). The model's label is not included, so it
cannot sway the annotator. The file holds post text, so it is refused inside
the repository.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from src.common import config
from src.ml.data import load_config
from src.stream.settings import mode_settings
from src.stream.sources.bluesky import PostClassifier
from src.stream.sources.filters import load_live_config, mask_handles

COLUMNS = ["sample_id", "text", "label"]


def inside_repo(path: str) -> bool:
    root = os.path.abspath(config.BASE_DIR)
    return os.path.abspath(path) == root or os.path.abspath(path).startswith(root + os.sep)


def choose(posts: pd.DataFrame, n: int, seed: int, warmup_minutes: float) -> pd.DataFrame:
    """`posts` has post_id, text, received_at_ms; returns n of those received after the warm-up."""
    start = posts["received_at_ms"].min()
    after = posts[posts["received_at_ms"] >= start + warmup_minutes * 60_000]
    after = after.sort_values("post_id").reset_index(drop=True)  # the draw does not depend on MongoDB's order
    if len(after) < n:
        raise SystemExit(f"Only {len(after)} posts after the warm-up; asked for {n}")
    picked = np.random.default_rng(seed).choice(len(after), size=n, replace=False)
    return after.iloc[picked].reset_index(drop=True)


def write_csv(sample: pd.DataFrame, path: str, keep_handles: set[str]) -> None:
    if inside_repo(path):
        raise SystemExit(f"Refusing to write post text inside the repository: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in sample.itertuples(index=False):
            writer.writerow([row.post_id, mask_handles(row.text, keep_handles), ""])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-minutes", type=float, default=5)
    parser.add_argument("--output", required=True, help="CSV path outside the repository")
    args = parser.parse_args(argv)
    if inside_repo(args.output):
        parser.error(f"--output must be outside the repository ({config.BASE_DIR})")

    from pymongo import MongoClient

    mongodb = mode_settings(load_config(), "live")["mongodb"]
    client = MongoClient(mongodb["uri"], serverSelectionTimeoutMS=10_000)
    documents = client[mongodb["database"]][mongodb["predictions_collection"]].find({}, {"text": 1, "received_at_ms": 1})
    posts = pd.DataFrame(
        [{"post_id": d["_id"], "text": d["text"], "received_at_ms": d["received_at_ms"]} for d in documents],
        columns=["post_id", "text", "received_at_ms"],
    )
    client.close()
    sample = choose(posts, args.n, args.seed, args.warmup_minutes)
    write_csv(sample, args.output, PostClassifier(load_live_config()).keep_handles)
    print(f"Wrote {len(sample)} posts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
