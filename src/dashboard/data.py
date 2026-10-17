"""Data behind the dashboard: a live MongoDB or recorded files, and the numbers shown.

The replay tab reads the airline tweets (MongoDB or reports/stream_snapshot/);
the live tab reads the Bluesky run (MongoDB or reports/live_run.json and
reports/live_snapshot/), counts and times only.

Nothing here imports Streamlit, so the calculations can be tested on their own.
"""
import json
import math
import os
from dataclasses import dataclass

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.common import config
from src.ml.data import LABELS

SOURCES = ("mongo", "snapshot", "auto")
WINDOW_COLUMNS = ["airline", "window_start", *LABELS, "total", "negative_share"]
LATEST_COLUMNS = ["created_at", "airline", "sentiment", "text"]
RECENT_COLUMNS = ["created_at", "airline", "sentiment"]


@dataclass
class Snapshot:
    predictions: pd.DataFrame  # ordered by created_at
    meta: dict


@dataclass
class View:
    """Everything one refresh of the page shows."""

    processed: int  # tweets with a prediction so far
    windows: pd.DataFrame  # WINDOW_COLUMNS, one row per airline and window
    recent: pd.DataFrame  # RECENT_COLUMNS, the tweets of the last two window lengths before `now`
    latest: pd.DataFrame  # LATEST_COLUMNS, newest first
    now: pd.Timestamp | None  # the event time the page is "at"; None before any data


def snapshot_dir(dashboard_cfg: dict, env=os.environ) -> str:
    path = env.get("SENTIMENT_SNAPSHOT_DIR") or dashboard_cfg["snapshot_dir"]
    return path if os.path.isabs(path) else os.path.join(config.BASE_DIR, path)


def load_snapshot(directory: str) -> Snapshot:
    predictions = pd.read_parquet(os.path.join(directory, "predictions.parquet"))
    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return Snapshot(predictions.sort_values("created_at", kind="stable").reset_index(drop=True), meta)


def mongo_reachable(uri: str, timeout_ms: int = 2000) -> bool:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        client.close()
        return True
    except PyMongoError:
        return False


def choose_source(requested: str, uri: str) -> str:
    """Resolve SENTIMENT_SOURCE to "mongo" or "snapshot"; `auto` tries MongoDB for 2 seconds."""
    if requested not in SOURCES:
        raise ValueError(f"SENTIMENT_SOURCE must be one of {', '.join(SOURCES)}, not {requested!r}")
    if requested == "auto":
        return "mongo" if mongo_reachable(uri) else "snapshot"
    return requested


def window_counts(predictions: pd.DataFrame, window: pd.Timedelta) -> pd.DataFrame:
    """Tweets per label for each airline and event-time window, like the Spark job's sink."""
    if predictions.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    counts = (
        predictions.assign(window_start=predictions["created_at"].dt.floor(window))
        .groupby(["airline", "window_start", "sentiment"])
        .size()
        .unstack("sentiment", fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    counts["total"] = counts[LABELS].sum(axis=1)
    counts["negative_share"] = counts["negative"] / counts["total"]
    return counts[WINDOW_COLUMNS].sort_values(["window_start", "airline"]).reset_index(drop=True)


def replay_view(snapshot: Snapshot, clock: pd.Timestamp, window: pd.Timedelta, latest: int) -> View:
    """The snapshot as it stood at `clock`: only tweets created at or before it."""
    visible = snapshot.predictions[snapshot.predictions["created_at"] <= clock]
    return View(
        processed=len(visible),
        windows=window_counts(visible, window),
        recent=visible.loc[visible["created_at"] > clock - 2 * window, RECENT_COLUMNS],
        latest=visible.tail(latest).iloc[::-1][LATEST_COLUMNS].reset_index(drop=True),
        now=clock,
    )


def live_view(database, mongodb_cfg: dict, window: pd.Timedelta, latest: int) -> View:
    """The current contents of the two collections the Spark job writes."""
    predictions = database[mongodb_cfg["predictions_collection"]]
    newest = list(predictions.find({}, dict.fromkeys(LATEST_COLUMNS, 1)).sort("created_at", -1).limit(latest))
    latest_frame = pd.DataFrame(newest, columns=LATEST_COLUMNS)
    window_frame = pd.DataFrame(
        list(database[mongodb_cfg["windows_collection"]].find({}, dict.fromkeys(WINDOW_COLUMNS, 1))),
        columns=WINDOW_COLUMNS,
    )
    # "Now" is the newest event time seen, not the wall clock: the stream replays old tweets.
    now = latest_frame["created_at"].max() if len(latest_frame) else None
    since = {"created_at": {"$gt": now - 2 * window}} if now is not None else {}
    recent_frame = pd.DataFrame(
        list(predictions.find(since, dict.fromkeys(RECENT_COLUMNS, 1))) if now is not None else [],
        columns=RECENT_COLUMNS,
    )
    for frame, column in (
        (latest_frame, "created_at"),
        (recent_frame, "created_at"),
        (window_frame, "window_start"),
    ):
        frame[column] = pd.to_datetime(frame[column]).dt.tz_localize("UTC")  # pymongo returns naive UTC
    return View(
        processed=predictions.estimated_document_count(),
        windows=window_frame.sort_values(["window_start", "airline"]).reset_index(drop=True),
        recent=recent_frame,
        latest=latest_frame,
        now=None if now is None else latest_frame["created_at"].max(),
    )


def last_hour_kpi(recent: pd.DataFrame, now: pd.Timestamp, window: pd.Timedelta, airlines=None) -> dict:
    """Negative share over the `window` ending at `now`, and over the `window` before that.

    A rolling hour, not the clock hour in progress: a clock hour that has only
    just started would be judged on a handful of tweets.
    """
    rows = recent if airlines is None else recent[recent["airline"].isin(airlines)]

    def share(after: pd.Timestamp, until: pd.Timestamp) -> tuple[float | None, int]:
        part = rows[(rows["created_at"] > after) & (rows["created_at"] <= until)]
        tweets = len(part)
        return (float((part["sentiment"] == "negative").sum() / tweets) if tweets else None), tweets

    current, tweets = share(now - window, now)
    previous, previous_tweets = share(now - 2 * window, now - window)
    return {
        "share": current,
        "tweets": tweets,
        "previous_share": previous,
        "previous_tweets": previous_tweets,
    }


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default). Sensible even for tiny n."""
    if n <= 0:
        raise ValueError("n must be positive")
    share = successes / n
    denominator = 1 + z * z / n
    centre = (share + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def current_window_table(
    windows: pd.DataFrame,
    now: pd.Timestamp,
    window: pd.Timedelta,
    threshold: float,
    min_tweets: int,
    airlines=None,
) -> pd.DataFrame:
    """One row per airline for the window containing `now`.

    `above_threshold` needs both enough tweets and a negative share over the
    threshold; a window with fewer than `min_tweets` tweets is never flagged.
    """
    rows = windows[windows["window_start"] == now.floor(window)]
    if airlines is not None:
        rows = rows[rows["airline"].isin(airlines)]
    table = rows[["airline", "total", "negative", "negative_share"]].copy()
    intervals = [wilson_interval(int(k), int(n)) for k, n in zip(table["negative"], table["total"])]
    table["ci_low"] = [low for low, _ in intervals]
    table["ci_high"] = [high for _, high in intervals]
    table["enough_tweets"] = table["total"] >= min_tweets
    table["above_threshold"] = table["enough_tweets"] & (table["negative_share"] > threshold)
    return table.sort_values("negative_share", ascending=False).reset_index(drop=True)


# --- the Bluesky live run ---------------------------------------------------------------
#
# Two sources, picked like the replay's: the run recorded by scripts/live_report.py
# (reports/live_run.json, reports/live_snapshot/), or MongoDB while `make live-up`
# runs. Only counts and times are read: never the text of a live post.

ALL_TOPICS = "all"  # the window row over every topic (src.stream.transform.ALL_TOPICS)
LIVE_WINDOW_COLUMNS = ["topic", "window_start", *LABELS, "total", "negative_share"]
LIVE_TIME_FIELDS = ["received_at_ms", "written_at", "created_at"]
LIVE_REPORT_PATH = os.path.join(config.BASE_DIR, "reports", "live_run.json")
LIVE_WINDOWS_PATH = os.path.join(config.BASE_DIR, "reports", "live_snapshot", "windows.parquet")
DOMAIN_SHIFT_PATH = os.path.join(config.BASE_DIR, "reports", "live_domain_shift.json")


@dataclass
class LiveHealth:
    """The pipeline-health numbers of the live tab; None where the source does not have them."""

    period: str  # what the rate and latencies cover, e.g. "the last 5 minutes"
    received: int | None  # posts the producer read (its log; only in the recorded report)
    written: int  # posts in live_predictions
    posts_per_second: float | None
    pipeline_median_s: float | None  # written_at - received_at
    pipeline_p95_s: float | None
    source_median_s: float | None  # received_at - created_at
    reconnects: int | None
    endpoint_switches: int | None
    count_check: dict  # predictions, window totals of `all` and of the topics, match
    last_received: pd.Timestamp | None = None


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_live_windows(path: str = LIVE_WINDOWS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)[LIVE_WINDOW_COLUMNS]


def choose_live_source(requested: str, has_live_data) -> str:
    """Resolve SENTIMENT_SOURCE for the live tab to "mongo" or "recorded".

    `snapshot` means the recorded run; `auto` uses MongoDB only when it holds
    live windows (`has_live_data()`), so a replay-only stack shows the recording.
    """
    if requested not in SOURCES:
        raise ValueError(f"SENTIMENT_SOURCE must be one of {', '.join(SOURCES)}, not {requested!r}")
    if requested == "auto":
        return "mongo" if has_live_data() else "recorded"
    return "mongo" if requested == "mongo" else "recorded"


def mongo_has_live_windows(uri: str, database: str, collection: str, timeout_ms: int = 2000) -> bool:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        found = client[database][collection].find_one({}, {"_id": 1}) is not None
        client.close()
        return found
    except PyMongoError:
        return False


def recording_label(run: dict) -> str:
    """Where the recorded numbers come from: the run's length, date and machine."""
    machine = run["machine"]
    where = "GitHub Actions runner" if machine.get("github_runner") else "local machine"
    started = pd.Timestamp(run["run"]["producer_started_at"])
    return (
        f"Recorded from one {run['run']['minutes']:.0f}-minute live run on {started:%Y-%m-%d %H:%M} UTC, "
        f"{where} ({machine['nproc']} vCPU, {machine['ram_total_gb']} GB RAM)"
    )


def recorded_health(run: dict) -> LiveHealth:
    """Pipeline health of the recorded run, after its warm-up (scripts/live_report.py)."""
    latency = run["latency_seconds"]["after_warmup"]
    check, timing = run["count_check"], run["run"]
    return LiveHealth(
        period=f"the {timing['measured_minutes']:.0f} minutes after the {timing['warmup_minutes']:.0f}-minute warm-up",
        received=run["posts"]["received_by_producer"],
        written=run["posts"]["written"],
        posts_per_second=run["rate"]["after_warmup"]["mean_posts_per_second"],
        pipeline_median_s=latency["pipeline"]["median_s"],
        pipeline_p95_s=latency["pipeline"]["p95_s"],
        source_median_s=latency["source"]["median_s"],
        reconnects=run["connection"]["reconnects"],
        endpoint_switches=run["connection"]["endpoint_switches"],
        count_check={
            "predictions": check["live_predictions"],
            "all": check["windows_total_all_rows"],
            "topics": check["windows_total_topic_rows"],
            "match": check["match"],
        },
    )


def live_count_check(predictions: int, windows: pd.DataFrame) -> dict:
    """Posts written against the window totals of `all` and of the topics."""
    is_all = windows["topic"] == ALL_TOPICS
    total_all, total_topics = int(windows.loc[is_all, "total"].sum()), int(windows.loc[~is_all, "total"].sum())
    return {
        "predictions": int(predictions),
        "all": total_all,
        "topics": total_topics,
        "match": int(predictions) == total_all == total_topics,
    }


def times_frame(documents) -> pd.DataFrame:
    """live_predictions documents (time fields only) -> received, written, created as UTC timestamps."""
    rows = [(d.get("received_at_ms"), d.get("written_at"), d.get("created_at")) for d in documents]
    frame = pd.DataFrame(rows, columns=LIVE_TIME_FIELDS)
    return pd.DataFrame(
        {
            "received": pd.to_datetime(frame["received_at_ms"], unit="ms", utc=True),
            # pymongo returns naive UTC datetimes
            "written": pd.to_datetime(frame["written_at"]).dt.tz_localize("UTC"),
            "created": pd.to_datetime(frame["created_at"]).dt.tz_localize("UTC"),
        }
    )


def recent_health(
    recent: pd.DataFrame,
    written: int,
    windows: pd.DataFrame,
    first_received: pd.Timestamp | None,
    minutes: int = 5,
) -> LiveHealth:
    """Pipeline health over the last `minutes` before the newest received post (MongoDB source).

    `recent` is times_frame() of the posts received in those minutes. The rate
    divides by the time covered, which is shorter at the start of a run.
    """
    period = f"the last {minutes} minutes"
    check = live_count_check(written, windows)
    if recent.empty:
        return LiveHealth(period, None, written, None, None, None, None, None, None, check)
    newest = recent["received"].max()
    start = newest - pd.Timedelta(minutes=minutes)
    if first_received is not None and first_received > start:
        start = first_received
    seconds = (newest - start).total_seconds()
    pipeline = (recent["written"] - recent["received"]).dt.total_seconds()
    source = (recent["received"] - recent["created"]).dt.total_seconds().dropna()
    return LiveHealth(
        period=period,
        received=None,
        written=written,
        posts_per_second=round(len(recent) / seconds, 2) if seconds > 0 else None,
        pipeline_median_s=round(float(pipeline.median()), 3),
        pipeline_p95_s=round(float(pipeline.quantile(0.95)), 3),
        source_median_s=round(float(source.median()), 3) if len(source) else None,
        reconnects=None,
        endpoint_switches=None,
        count_check=check,
        last_received=newest,
    )


def per_minute(windows: pd.DataFrame) -> pd.DataFrame:
    """Posts and negative share per 1-minute window, over every topic (the `all` rows)."""
    rows = windows[windows["topic"] == ALL_TOPICS]
    columns = ["window_start", "total", "negative", "negative_share"]
    return rows[columns].sort_values("window_start").reset_index(drop=True)


def topic_counts(windows: pd.DataFrame) -> pd.DataFrame:
    """Posts per topic: general, air_travel, then each airline:<name>. Counts only: too sparse to chart."""
    totals = windows[windows["topic"] != ALL_TOPICS].groupby("topic")["total"].sum()
    airlines = sorted(t for t in totals.index if t.startswith("airline:"))
    order = ["general", "air_travel", *airlines]
    return pd.DataFrame({"topic": order, "posts": [int(totals.get(t, 0)) for t in order]})


def domain_shift_summary(shift: dict) -> dict:
    """The numbers of the out-of-domain warning, from reports/live_domain_shift.json."""
    reference = shift["vs_2015_test_set"]
    return {
        "n": shift["n"],
        "skipped": shift["rows"]["skip"],
        "macro_f1": shift["model"]["macro_f1"],
        "macro_f1_ci95": shift["model"]["macro_f1_ci95"],
        "macro_f1_2015": reference["macro_f1_2015"],
        "macro_f1_ci95_2015": reference["macro_f1_ci95_2015"],
        "model_negative": shift["model_label_counts"]["negative"],
        "human_negative": shift["hand_label_counts"]["negative"],
    }


def live_mongo_data(database, mongodb_cfg: dict, minutes: int = 5):
    """(times of the last `minutes` of posts, posts written, windows, first received) from MongoDB.

    Reads only the time fields of live_predictions, never the text.
    """
    predictions = database[mongodb_cfg["predictions_collection"]]
    newest = predictions.find_one({}, {"received_at_ms": 1}, sort=[("received_at_ms", -1)])
    oldest = predictions.find_one({}, {"received_at_ms": 1}, sort=[("received_at_ms", 1)])
    recent_docs = []
    if newest is not None:
        since = newest["received_at_ms"] - minutes * 60_000
        fields = {"_id": 0, **dict.fromkeys(LIVE_TIME_FIELDS, 1)}  # never the text
        recent_docs = list(predictions.find({"received_at_ms": {"$gte": since}}, fields))
    windows = pd.DataFrame(
        list(database[mongodb_cfg["windows_collection"]].find({}, dict.fromkeys(LIVE_WINDOW_COLUMNS, 1))),
        columns=LIVE_WINDOW_COLUMNS,
    )
    windows["window_start"] = pd.to_datetime(windows["window_start"]).dt.tz_localize("UTC")
    first = None if oldest is None else pd.Timestamp(oldest["received_at_ms"], unit="ms", tz="UTC")
    return times_frame(recent_docs), predictions.count_documents({}), windows, first
