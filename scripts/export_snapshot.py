"""Record one real run of the stack as a small snapshot for the dashboard's replay mode.

Usage (from the repo root, on a fresh stack):
    docker compose down -v
    docker compose up -d
    python -m scripts.export_snapshot

The script replays the whole test set through Kafka and Spark (`producer
--speedup 600`), waits until MongoDB holds every prediction and every window
count, and writes reports/stream_snapshot/:

  predictions.parquet  one row per tweet, with the predicted sentiment
  windows.parquet      the airline_windows collection as Spark wrote it
  meta.json            when and from which commit the run was recorded, and its counts
  README.md            the data license

It refuses to start if MongoDB already holds data, and writes nothing unless
the number of predictions and the sum of the window totals both equal the
number of test tweets. The snapshot contains tweet text: CC BY-NC-SA 4.0.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import timezone

import pandas as pd
from pymongo import MongoClient

from scripts.benchmark_report import REPO_ROOT, commit_sha, utc_now
from scripts.benchmark_stream import docker_info, processor_log_events, processor_versions, run
from src.ml.data import load_config, load_raw
from src.stream.producer import select_test_tweets
from src.stream.settings import stream_settings

SNAPSHOT_DIR = os.path.join(REPO_ROOT, "reports", "stream_snapshot")
MAX_BYTES = 5 * 1024 * 1024
PREDICTION_COLUMNS = [
    "tweet_id",
    "airline",
    "text",
    "created_at",
    "sentiment",
    "produced_at_ms",
    "written_at",
]
WINDOW_COLUMNS = [
    "_id",
    "airline",
    "window_start",
    "window_end",
    "negative",
    "neutral",
    "positive",
    "total",
    "negative_share",
    "updated_at",
]
LICENSE_NOTE = """# Stream snapshot

Recorded by `python -m scripts.export_snapshot`; see `meta.json` for the run.
These files contain tweet text from the Twitter US Airline Sentiment dataset and
are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(see `data/raw/SOURCE.md`), not under the repository's MIT license.
"""


def as_utc(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """pymongo returns naive datetimes in UTC; make the time zone explicit."""
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).dt.tz_localize(timezone.utc)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speedup", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for MongoDB after the replay")
    parser.add_argument("--output", default=SNAPSHOT_DIR)
    args = parser.parse_args()

    cfg = load_config()
    settings = stream_settings(cfg)
    mongodb = settings["mongodb"]
    client = MongoClient(mongodb["uri"], serverSelectionTimeoutMS=10_000)
    database = client[mongodb["database"]]
    predictions = database[mongodb["predictions_collection"]]
    windows = database[mongodb["windows_collection"]]
    if predictions.estimated_document_count() or windows.estimated_document_count():
        print("MongoDB already holds data. Start from a fresh stack: docker compose down -v, then up -d.")
        return 1

    expected = len(select_test_tweets(load_raw(), cfg))
    command = ["docker", "compose", "run", "--rm", "producer", "--speedup", str(args.speedup)]
    print(f"Replaying {expected} test tweets: {' '.join(command)}")
    produced = re.search(r"sent=(\d+) failed=(\d+)", run(command))
    sent, failed = (int(produced.group(1)), int(produced.group(2))) if produced else (0, 0)
    if sent != expected or failed:
        print(f"The producer sent {sent} of {expected} tweets ({failed} failed); nothing written.")
        return 1

    def window_total() -> int:
        grouped = windows.aggregate([{"$group": {"_id": None, "total": {"$sum": "$total"}}}])
        return next((int(row["total"]) for row in grouped), 0)

    deadline = time.monotonic() + args.timeout
    while predictions.count_documents({}) < expected or window_total() < expected:
        if time.monotonic() > deadline:
            print(
                f"Timed out with {predictions.count_documents({})} predictions and a window total "
                f"of {window_total()} (expected {expected}); nothing written."
            )
            return 1
        time.sleep(2)
    time.sleep(10)  # let the last progress events reach the processor's log

    prediction_frame = pd.DataFrame(list(predictions.find({}))).rename(columns={"_id": "tweet_id"})
    prediction_frame = as_utc(prediction_frame, ["created_at", "written_at"])
    prediction_frame = prediction_frame[PREDICTION_COLUMNS].sort_values(["created_at", "tweet_id"])
    window_frame = as_utc(pd.DataFrame(list(windows.find({}))), ["window_start", "window_end", "updated_at"])
    window_frame = window_frame[WINDOW_COLUMNS].sort_values(["window_start", "airline"])

    total = int(window_frame["total"].sum())
    if len(prediction_frame) != expected or total != expected or not prediction_frame["tweet_id"].is_unique:
        print(f"Inconsistent run: {len(prediction_frame)} predictions, window total {total}, expected {expected}.")
        return 1

    progress, _ = processor_log_events()
    window_batches = [p for p in progress if p["name"] == "airline_windows"]
    meta = {
        "recorded_at_utc": utc_now(),
        "commit": commit_sha(),
        "command": " ".join(command),
        "speedup": args.speedup,
        "tweets_sent": sent,
        "records": {
            mongodb["predictions_collection"]: len(prediction_frame),
            mongodb["windows_collection"]: len(window_frame),
        },
        "window_total": total,
        "rows_dropped_by_watermark": sum(p["rowsDroppedByWatermark"] for p in window_batches),
        "duplicates_dropped": sum(p["duplicatesDropped"] for p in window_batches),
        "first_created_at": prediction_frame["created_at"].min().isoformat(),
        "last_created_at": prediction_frame["created_at"].max().isoformat(),
        "settings": {key: settings[key] for key in ("trigger", "window", "watermark")},
        "images": docker_info()["images"],
        "versions": {**processor_versions(), "mongodb_server": client.server_info()["version"]},
        "license": "CC BY-NC-SA 4.0 (contains tweet text; see data/raw/SOURCE.md)",
    }

    os.makedirs(args.output, exist_ok=True)
    prediction_frame.to_parquet(os.path.join(args.output, "predictions.parquet"), index=False)
    window_frame.to_parquet(os.path.join(args.output, "windows.parquet"), index=False)
    with open(os.path.join(args.output, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    with open(os.path.join(args.output, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(LICENSE_NOTE)

    size = sum(os.path.getsize(os.path.join(args.output, name)) for name in os.listdir(args.output))
    print(f"Wrote {args.output}: {len(prediction_frame)} predictions, {len(window_frame)} windows, {size} bytes")
    if size > MAX_BYTES:
        print(f"The snapshot is larger than {MAX_BYTES} bytes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
