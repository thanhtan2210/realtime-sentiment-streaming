"""Statistics of scripts/live_report.py on invented posts and log lines (no MongoDB, no Docker)."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.live_report import (
    MODEL_NOTE,
    build_report,
    count_check,
    distribution,
    ingest_totals,
    latencies,
    log_events,
    predictions_frame,
    processor_totals,
    rates,
    stats,
)

T0 = 1_800_000_000.0
MACHINE = {"nproc": 2}


def posts_frame(n=600, step=1.0, source_lag=2.0, pipeline=4.0):
    """n posts, one every `step` seconds, each `source_lag` s old when received and written `pipeline` s later."""
    received = T0 + np.arange(n) * step
    return pd.DataFrame(
        {
            "topic": ["general"] * (n - 2) + ["air_travel", "airline:United"],
            "sentiment": (["negative", "neutral", "positive"] * n)[:n],
            "created": received - source_lag,
            "event": received - source_lag + 0.1,
            "received": received,
            "written": received + pipeline,
        }
    )


def windows_frame(posts):
    rows = []
    for topic in [*posts["topic"].unique(), "all"]:
        part = posts if topic == "all" else posts[posts["topic"] == topic]
        minute = (part["event"] // 60) * 60
        for start, group in part.groupby(minute):
            counts = group["sentiment"].value_counts()
            neg, neu, pos = (int(counts.get(k, 0)) for k in ("negative", "neutral", "positive"))
            rows.append({"topic": topic, "window_start": start, "window_end": start + 60, "negative": neg,
                         "neutral": neu, "positive": pos, "total": len(group), "negative_share": neg / len(group)})  # fmt: skip
    return pd.DataFrame(rows)


def test_stats_percentiles_and_missing_values():
    result = stats([1, 2, 3, 4, np.nan, 100])
    assert result["n"] == 5 and result["median_s"] == 3.0
    assert result["p95_s"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95), abs=1e-3)
    assert stats([])["median_s"] is None


def test_three_latencies_and_the_clock_offset():
    posts = posts_frame(source_lag=2.0, pipeline=4.0)
    result = latencies(posts, offset=0.5)
    assert result["source"]["median_s"] == pytest.approx(2.5)  # received + offset - created
    assert result["pipeline"]["median_s"] == pytest.approx(4.0)  # the offset cancels out
    assert result["end_to_end"]["median_s"] == pytest.approx(6.5)
    posts.loc[0, "created"] = np.nan
    assert latencies(posts, 0.0)["source"]["n"] == len(posts) - 1


def test_rates_by_full_minute():
    received = T0 + np.arange(0, 300, 0.5)  # 2 posts per second for 5 minutes
    result = rates(received, T0, T0 + 300)
    assert result["full_minutes"] == 5 and result["posts"] == 600
    assert result["mean_posts_per_second"] == 2.0
    assert result["median_posts_per_second_by_minute"] == 2.0 and result["p95_posts_per_second_by_minute"] == 2.0
    partial = rates(received, T0, T0 + 90)  # 1.5 minutes: one full minute
    assert partial["full_minutes"] == 1 and partial["posts"] == 180


def test_count_check_matches_and_detects_a_difference():
    posts = posts_frame()
    windows = windows_frame(posts)
    ok = count_check(len(posts), windows)
    assert ok["match"] and ok["windows_total_all_rows"] == ok["windows_total_topic_rows"] == len(posts)
    assert ok["windows_where_all_differs_from_topics"] == 0 and ok["label_counts_add_up"]
    windows.loc[windows["topic"] == "all", "total"] += 1
    broken = count_check(len(posts), windows)
    assert not broken["match"] and broken["windows_where_all_differs_from_topics"] == ok["windows"]


def test_distribution_keeps_the_given_order_and_zero_counts():
    result = distribution(pd.Series(["negative", "negative", "positive"]), order=["negative", "neutral", "positive"])
    assert list(result) == ["negative", "neutral", "positive"]
    assert result["neutral"] == {"posts": 0, "share": 0.0}
    assert result["negative"]["share"] == pytest.approx(2 / 3, abs=1e-5)


def test_log_lines_and_totals_across_a_producer_restart():
    switch = {"from": "a.example", "to": "b.example", "reason": "slow", "after_seconds": 120.0}
    first = {"started_at": "s1", "stopped_at": "e1", "received": 100, "sent": 100, "failed": 0, "connections": 2,
             "dropped_connections": 0, "failed_connection_attempts": 1, "endpoint_switches": 1, "switches": [switch],
             "stopped_by": "signal", "clock_offset_start_s": 0.5, "clock_offset_end_s": 0.7}  # fmt: skip
    second = {**first, "started_at": "s2", "stopped_at": "e2", "received": 50, "sent": 50, "connections": 1,
              "endpoint_switches": 0, "switches": [], "clock_offset_end_s": None}  # fmt: skip
    lines = ["noise", "2026-01-01 - INFO - LIVE_SUMMARY " + json.dumps(first), "LIVE_SUMMARY " + json.dumps(second)]
    totals = ingest_totals(log_events(lines, "LIVE_SUMMARY"))
    assert totals["summaries"] == 2 and totals["received"] == 150
    assert (totals["started_at"], totals["stopped_at"]) == ("s1", "e2")
    assert totals["connections"] == 3 and totals["reconnects"] == 1  # one connection per start is not a reconnect
    assert totals["endpoint_switches"] == 1 and totals["switches"] == [switch]
    assert totals["clock_offsets_s"] == [0.5, 0.7, 0.5]
    assert ingest_totals([]) == {"summaries": 0}


def test_processor_totals_only_count_the_predictions_query():
    progress = [
        {"name": "live_predictions", "numInputRows": 10, "duplicatesDropped": 2, "rowsDroppedByWatermark": 0},
        {"name": "live_windows", "numInputRows": 10, "duplicatesDropped": 2, "rowsDroppedByWatermark": 0},
        {"name": "live_predictions", "numInputRows": 5, "duplicatesDropped": 0, "rowsDroppedByWatermark": 1},
    ]
    assert processor_totals(progress, "live_predictions") == {
        "batches_with_input": 2,
        "rows_read_from_kafka": 15,
        "dropped_as_duplicate": 2,
        "dropped_as_late": 1,
    }


def test_predictions_frame_reads_naive_utc_datetimes():
    written = datetime(2027, 1, 15, 8, 0, 5)  # MongoDB returns naive UTC datetimes
    documents = [{"topic": "general", "sentiment": "neutral", "created_at": None, "event_time": written,
                  "received_at_ms": 1_800_000_000_000, "written_at": written}]  # fmt: skip
    frame = predictions_frame(documents)
    assert np.isnan(frame.loc[0, "created"])
    assert frame.loc[0, "written"] == datetime(2027, 1, 15, 8, 0, 5, tzinfo=timezone.utc).timestamp()
    assert frame.loc[0, "received"] == 1_800_000_000.0


def test_report_splits_warmup_and_has_only_aggregates():
    posts = posts_frame(n=600)  # 10 minutes, one post a second
    progress = [{"name": "live_predictions", "numInputRows": 610, "duplicatesDropped": 10, "rowsDroppedByWatermark": 0}]
    summary = {"started_at": "s", "stopped_at": "e", "received": 610, "sent": 610, "failed": 0, "connections": 1,
               "dropped_connections": 0, "failed_connection_attempts": 0, "endpoint_switches": 0, "switches": [],
               "stopped_by": "signal", "clock_offset_start_s": 0.0, "clock_offset_end_s": 0.0}  # fmt: skip
    report = build_report(posts, windows_frame(posts), [summary], progress, 5, 0.0, "test", MACHINE, "live_predictions")

    assert report["rate"]["warmup"]["posts"] == 300 and report["rate"]["after_warmup"]["posts"] == 300
    assert report["rate"]["after_warmup"]["mean_posts_per_second"] == pytest.approx(1.0, abs=0.01)
    assert report["latency_seconds"]["after_warmup"]["pipeline"]["n"] == 300
    assert report["posts"]["written"] == 600 and report["posts"]["written_equals_read_minus_dropped"]
    assert report["count_check"]["match"]
    assert report["model_labels"]["note"] == MODEL_NOTE
    assert report["topics"]["airline:United"]["posts"] == 1
    # nothing but numbers, times, topic names and endpoint hosts
    blob = json.dumps(report)
    assert "did:" not in blob and "text" not in blob
