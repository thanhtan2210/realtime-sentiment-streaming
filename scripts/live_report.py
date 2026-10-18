"""Report on one live-mode run; writes reports/live_run.json and reports/live_snapshot/windows.parquet.

Usage (from the repo root, after `docker compose stop ingest-live` and once the
processor has written everything):
    python -m scripts.live_report --warmup-minutes 5

Reads MongoDB (live_predictions without the text, live_windows) and the
`LIVE_SUMMARY` / `PROGRESS` lines of the ingest-live and processor-live logs
(`docker compose logs`, or files given with --ingest-log / --processor-log).
Everything written is aggregate: counts, rates, percentiles and 1-minute window
counts. No post text, handle or DID.

The run starts at the first post the producer received; the first
--warmup-minutes are reported apart from the rest. Latencies, in seconds:

  source      received_at - created_at   (Bluesky to the producer)
  pipeline    written_at - received_at   (producer -> Kafka -> Spark -> MongoDB)
  end_to_end  written_at - created_at

created_at is the post's own createdAt, set by the author's app; received_at
and written_at come from the containers' clock (one Docker VM), corrected by
the SNTP offset the producer measured, which cancels out of `pipeline`.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.benchmark_report import commit_sha, machine_info, utc_now
from src.common import config
from src.ml.data import LABELS, load_config
from src.stream.settings import mode_settings
from src.stream.sources.filters import clock_offset, load_live_config

OUTPUT_PATH = os.path.join(config.BASE_DIR, "reports", "live_run.json")
SNAPSHOT_DIR = os.path.join(config.BASE_DIR, "reports", "live_snapshot")
# Same as src.stream.transform.ALL_TOPICS, which is not imported here because it
# needs pyspark (this script runs outside the Spark image).
ALL_TOPICS = "all"
MODEL_NOTE = "model trained on 2015 airline tweets, not validated on this data"
PREDICTION_FIELDS = ["topic", "sentiment", "created_at", "event_time", "received_at_ms", "written_at"]
WINDOW_FIELDS = ["topic", "window_start", "window_end", *LABELS, "total", "negative_share"]


def epoch(value) -> float:
    """Seconds since the epoch; MongoDB hands datetimes back without a time zone (they are UTC)."""
    if value is None:
        return np.nan
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def iso(seconds: float | None) -> str | None:
    if seconds is None or np.isnan(seconds):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="seconds")


def predictions_frame(documents) -> pd.DataFrame:
    """Prediction documents -> one row per post, times in epoch seconds."""
    rows = [
        {
            "topic": d["topic"],
            "sentiment": d["sentiment"],
            "created": epoch(d.get("created_at")),
            "event": epoch(d["event_time"]),
            "received": d["received_at_ms"] / 1000,
            "written": epoch(d["written_at"]),
        }
        for d in documents
    ]
    return pd.DataFrame(rows, columns=["topic", "sentiment", "created", "event", "received", "written"])


def log_events(lines, marker: str) -> list[dict]:
    """The JSON after `marker` on each log line that has it."""
    events = []
    for line in lines:
        if marker + " {" in line:
            events.append(json.loads(line.split(marker + " ", 1)[1]))
    return events


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"n": 0, "median_s": None, "p95_s": None, "p99_s": None}
    median, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"n": int(values.size), "median_s": round(float(median), 3), "p95_s": round(float(p95), 3), "p99_s": round(float(p99), 3)}


def latencies(posts: pd.DataFrame, offset: float) -> dict:
    received, written = posts["received"] + offset, posts["written"] + offset
    return {
        "source": stats(received - posts["created"]),
        "pipeline": stats(posts["written"] - posts["received"]),
        "end_to_end": stats(written - posts["created"]),
    }


def rates(received: np.ndarray, start: float, end: float) -> dict:
    """Posts per second over [start, end): the mean, and the median and p95 over its full minutes."""
    received = np.asarray(received, dtype=float)
    inside = received[(received >= start) & (received < end)]
    seconds = end - start
    full_minutes = int(seconds // 60)
    per_minute = np.bincount(((inside - start) // 60).astype(int), minlength=full_minutes)[:full_minutes] / 60
    return {
        "seconds": round(seconds, 1),
        "posts": int(inside.size),
        "mean_posts_per_second": round(inside.size / seconds, 3) if seconds > 0 else None,
        "full_minutes": full_minutes,
        "median_posts_per_second_by_minute": round(float(np.median(per_minute)), 3) if full_minutes else None,
        "p95_posts_per_second_by_minute": round(float(np.percentile(per_minute, 95)), 3) if full_minutes else None,
    }


def distribution(values: pd.Series, order=None) -> dict:
    counts = values.value_counts()
    keys = order if order is not None else list(counts.index)
    total = int(counts.sum())
    return {k: {"posts": int(counts.get(k, 0)), "share": round(int(counts.get(k, 0)) / total, 5) if total else None} for k in keys}


def count_check(predictions: int, windows: pd.DataFrame) -> dict:
    all_rows = windows[windows["topic"] == ALL_TOPICS]
    topic_rows = windows[windows["topic"] != ALL_TOPICS]
    per_window = topic_rows.groupby("window_start")["total"].sum()
    joined = all_rows.set_index("window_start")["total"].to_frame("all").join(per_window.rename("topics"), how="outer").fillna(0)
    total_all, total_topics = int(all_rows["total"].sum()), int(topic_rows["total"].sum())
    return {
        "live_predictions": predictions,
        "windows_total_all_rows": total_all,
        "windows_total_topic_rows": total_topics,
        "windows": int(len(all_rows)),
        "windows_where_all_differs_from_topics": int((joined["all"] != joined["topics"]).sum()),
        "label_counts_add_up": bool((windows[LABELS].sum(axis=1) == windows["total"]).all()),
        "match": predictions == total_all == total_topics,
    }


def ingest_totals(summaries: list[dict]) -> dict:
    """LIVE_SUMMARY lines of the producer (more than one if it was restarted), added up."""
    if not summaries:
        return {"summaries": 0}
    added = {
        key: sum(s.get(key, 0) for s in summaries)
        for key in ("received", "sent", "failed", "connections", "dropped_connections", "failed_connection_attempts", "endpoint_switches")
    }
    offsets = [s.get(k) for s in summaries for k in ("clock_offset_start_s", "clock_offset_end_s")]
    offsets = [o for o in offsets if o is not None]
    return {
        "summaries": len(summaries),
        "started_at": summaries[0].get("started_at"),
        "stopped_at": summaries[-1].get("stopped_at"),
        "stopped_by": [s.get("stopped_by") for s in summaries],
        **added,
        "reconnects": added["connections"] - len(summaries),
        "switches": [switch for s in summaries for switch in s.get("switches", [])],
        "clock_offsets_s": offsets,
    }


def processor_totals(progress: list[dict], query: str) -> dict:
    batches = [p for p in progress if p.get("name") == query]
    return {
        "batches_with_input": len(batches),
        "rows_read_from_kafka": sum(p["numInputRows"] for p in batches),
        "dropped_as_duplicate": sum(p.get("duplicatesDropped", 0) for p in batches),
        "dropped_as_late": sum(p.get("rowsDroppedByWatermark", 0) for p in batches),
    }


def build_report(posts, windows, summaries, progress, warmup_minutes, offset, offset_source, machine, query) -> dict:
    ingest = ingest_totals(summaries)
    processor = processor_totals(progress, query)
    start, end = float(posts["received"].min()), float(posts["received"].max())
    last = end + 1e-3  # the ranges below leave out their end; keep the last post in
    warmup_end = start + warmup_minutes * 60
    warmup, measured = posts[posts["received"] < warmup_end], posts[posts["received"] >= warmup_end]
    written = int(len(posts))
    return {
        "generated_at": utc_now(),
        "commit": commit_sha(),
        "machine": machine,
        "clock": {"offset_used_s": offset, "offset_source": offset_source, "producer_offsets_s": ingest.get("clock_offsets_s")},
        "run": {
            "producer_started_at": ingest.get("started_at"),
            "producer_stopped_at": ingest.get("stopped_at"),
            "first_post_received_at": iso(start),
            "last_post_received_at": iso(end),
            "minutes": round((end - start) / 60, 2),
            "warmup_minutes": warmup_minutes,
            "measured_minutes": round(max(end - warmup_end, 0) / 60, 2),
        },
        "posts": {
            "received_by_producer": ingest.get("received"),
            "sent_to_kafka": ingest.get("sent"),
            "failed_kafka_sends": ingest.get("failed"),
            **processor,
            "written": written,
            "written_equals_read_minus_dropped": processor["rows_read_from_kafka"] - processor["dropped_as_duplicate"]
            - processor["dropped_as_late"] == written,
        },
        "rate": {"warmup": rates(posts["received"], start, warmup_end), "after_warmup": rates(posts["received"], warmup_end, last)},
        "latency_seconds": {
            "definitions": {
                "source": "received_at - created_at",
                "pipeline": "written_at - received_at",
                "end_to_end": "written_at - created_at",
            },
            "warmup": latencies(warmup, offset),
            "after_warmup": latencies(measured, offset),
            "posts_without_created_at": int(posts["created"].isna().sum()),
        },
        "connection": {
            key: ingest.get(key)
            for key in ("connections", "reconnects", "dropped_connections", "failed_connection_attempts", "endpoint_switches", "switches", "stopped_by")
        },
        "count_check": count_check(written, windows),
        "topics": distribution(posts["topic"]),
        "model_labels": {"note": MODEL_NOTE, "counts": distribution(posts["sentiment"], order=LABELS)},
    }


def docker_logs(service: str) -> list[str]:
    result = subprocess.run(
        ["docker", "compose", "--profile", "live", "logs", "--no-log-prefix", "--no-color", service],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    return (result.stdout + result.stderr).splitlines()


def read_lines(path: str | None, service: str) -> list[str]:
    if path:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    return docker_logs(service)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup-minutes", type=float, default=5)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--snapshot-dir", default=SNAPSHOT_DIR)
    parser.add_argument("--ingest-log", help="a saved ingest-live log instead of `docker compose logs`")
    parser.add_argument("--processor-log", help="a saved processor-live log instead of `docker compose logs`")
    args = parser.parse_args(argv)

    from pymongo import MongoClient

    settings = mode_settings(load_config(), "live")
    mongodb = settings["mongodb"]
    client = MongoClient(mongodb["uri"], serverSelectionTimeoutMS=10_000)
    database = client[mongodb["database"]]
    documents = database[mongodb["predictions_collection"]].find({}, {field: 1 for field in PREDICTION_FIELDS})
    posts = predictions_frame(documents)
    windows = pd.DataFrame(
        list(database[mongodb["windows_collection"]].find({}, {"_id": 0, **{field: 1 for field in WINDOW_FIELDS}})),
        columns=WINDOW_FIELDS,
    )
    client.close()
    if posts.empty:
        raise SystemExit("live_predictions is empty: nothing to report")

    summaries = log_events(read_lines(args.ingest_log, "ingest-live"), "LIVE_SUMMARY")
    progress = log_events(read_lines(args.processor_log, "processor-live"), "PROGRESS")
    producer_offsets = ingest_totals(summaries).get("clock_offsets_s") or []
    if producer_offsets:
        offset, source = round(float(np.mean(producer_offsets)), 3), "SNTP from the ingest-live container, start and end"
    else:
        host = clock_offset(load_live_config()["bluesky"]["time_server"])
        offset, source = (host, "SNTP from this machine") if host is not None else (0.0, "none measured")

    report = build_report(
        posts, windows, summaries, progress, args.warmup_minutes, offset, source, machine_info(),
        query=mongodb["predictions_collection"],
    )  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    os.makedirs(args.snapshot_dir, exist_ok=True)
    snapshot = windows.copy()
    for column in ("window_start", "window_end"):
        snapshot[column] = pd.to_datetime(snapshot[column], utc=True)
    snapshot.sort_values(["window_start", "topic"]).reset_index(drop=True).to_parquet(
        os.path.join(args.snapshot_dir, "windows.parquet"), index=False
    )
    check = report["count_check"]
    print(
        f"Wrote {args.output}: {report['posts']['written']} posts, count check "
        f"{'OK' if check['match'] else 'MISMATCH'}, {report['connection']['endpoint_switches']} endpoint switches"
    )
    return 0 if check["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
