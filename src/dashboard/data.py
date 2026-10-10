"""Data behind the dashboard: a live MongoDB or a recorded snapshot, and the numbers shown.

Nothing here imports Streamlit, so the calculations can be tested on their own.
"""
import json
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


def current_window_table(
    windows: pd.DataFrame, now: pd.Timestamp, window: pd.Timedelta, threshold: float, airlines=None
) -> pd.DataFrame:
    """One row per airline for the window containing `now`, flagged when above the threshold."""
    rows = windows[windows["window_start"] == now.floor(window)]
    if airlines is not None:
        rows = rows[rows["airline"].isin(airlines)]
    table = rows[["airline", "total", "negative", "negative_share"]].copy()
    table["above_threshold"] = table["negative_share"] > threshold
    return table.sort_values("negative_share", ascending=False).reset_index(drop=True)
