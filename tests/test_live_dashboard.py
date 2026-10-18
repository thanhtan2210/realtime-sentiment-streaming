"""The dashboard's Bluesky live tab: its calculations on invented data, and the page on the recorded run.

No MongoDB and no network: the page tests use SENTIMENT_SOURCE=snapshot.
"""
import base64
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.common import config
from src.dashboard import data
from src.dashboard.app import LIVE_MONGO_BANNER, out_of_domain_text

T0 = pd.Timestamp("2027-01-15T08:00:00Z")


def invented_windows() -> pd.DataFrame:
    """Two minutes: 10 then 4 posts in `all`, split over the topics."""
    rows = [
        ("all", 0, 6, 3, 1), ("general", 0, 5, 3, 1), ("air_travel", 0, 1, 0, 0),
        ("all", 1, 1, 2, 1), ("general", 1, 0, 2, 1), ("airline:United", 1, 1, 0, 0),
    ]  # fmt: skip
    frame = pd.DataFrame(
        [
            {"topic": topic, "window_start": T0 + pd.Timedelta(minutes=m), "negative": neg, "neutral": neu,
             "positive": pos, "total": neg + neu + pos, "negative_share": neg / (neg + neu + pos)}
            for topic, m, neg, neu, pos in rows
        ]
    )  # fmt: skip
    return frame[data.LIVE_WINDOW_COLUMNS]


def invented_run() -> dict:
    return {
        "machine": {"nproc": 4, "ram_total_gb": 15.6, "github_runner": True},
        "run": {"producer_started_at": "2027-01-15T08:00:01.000+00:00", "minutes": 45.2, "warmup_minutes": 5.0,
                "measured_minutes": 40.2},
        "posts": {"received_by_producer": 120, "written": 118},
        "rate": {"after_warmup": {"mean_posts_per_second": 11.5}},
        "latency_seconds": {
            "after_warmup": {
                "pipeline": {"median_s": 3.4, "p95_s": 5.6},
                "source": {"median_s": 1.1, "p95_s": 9.8},
            }
        },
        "connection": {"reconnects": 1, "endpoint_switches": 2},
        "count_check": {"live_predictions": 118, "windows_total_all_rows": 118, "windows_total_topic_rows": 118, "match": True},
    }


def invented_shift() -> dict:
    return {
        "n": 50,
        "rows": {"rows": 60, "skip": 10, "empty": 0, "scored": 50},
        "hand_label_counts": {"negative": 10, "neutral": 20, "positive": 20},
        "model_label_counts": {"negative": 30, "neutral": 10, "positive": 10},
        "model": {"macro_f1": 0.4, "macro_f1_ci95": [0.3, 0.5]},
        "vs_2015_test_set": {"macro_f1_2015": 0.7, "macro_f1_ci95_2015": [0.68, 0.72]},
    }


# --- the calculations -------------------------------------------------------------------


def test_live_source_choice():
    assert data.choose_live_source("snapshot", lambda: True) == "recorded"
    assert data.choose_live_source("mongo", lambda: False) == "mongo"
    assert data.choose_live_source("auto", lambda: True) == "mongo"
    assert data.choose_live_source("auto", lambda: False) == "recorded"
    with pytest.raises(ValueError):
        data.choose_live_source("kafka", lambda: True)


def test_nothing_listening_means_no_live_windows():
    assert data.mongo_has_live_windows("mongodb://127.0.0.1:9", "sentiment", "live_windows", timeout_ms=300) is False


def test_recorded_health_and_label_come_from_the_report():
    run = invented_run()
    health = data.recorded_health(run)
    assert (health.received, health.written, health.posts_per_second) == (120, 118, 11.5)
    assert (health.pipeline_median_s, health.pipeline_p95_s, health.source_median_s) == (3.4, 5.6, 1.1)
    assert (health.reconnects, health.endpoint_switches) == (1, 2)
    assert health.count_check == {"predictions": 118, "all": 118, "topics": 118, "match": True}
    assert health.period == "the 40 minutes after the 5-minute warm-up" and health.last_received is None
    assert data.recording_label(run) == (
        "Recorded from one 45-minute live run on 2027-01-15 08:00 UTC, GitHub Actions runner (4 vCPU, 15.6 GB RAM)"
    )


def test_count_check_on_windows():
    windows = invented_windows()
    assert data.live_count_check(14, windows) == {"predictions": 14, "all": 14, "topics": 14, "match": True}
    assert data.live_count_check(15, windows)["match"] is False


def test_per_minute_uses_the_all_rows_in_time_order():
    minutes = data.per_minute(invented_windows().iloc[::-1])
    assert minutes["window_start"].tolist() == [T0, T0 + pd.Timedelta(minutes=1)]
    assert minutes["total"].tolist() == [10, 4]
    assert minutes["negative_share"].tolist() == pytest.approx([0.6, 0.25])


def test_topic_counts_lists_general_air_travel_then_airlines():
    counts = data.topic_counts(invented_windows())
    assert counts.values.tolist() == [["general", 12], ["air_travel", 1], ["airline:United", 1]]
    empty = data.topic_counts(invented_windows().query("topic == 'all'"))
    assert empty.values.tolist() == [["general", 0], ["air_travel", 0]]  # always shown, even at zero


def times(received_seconds, pipeline=4.0, source=2.0) -> pd.DataFrame:
    received = pd.to_datetime([T0 + pd.Timedelta(seconds=s) for s in received_seconds])
    return pd.DataFrame({"received": received, "written": received + pd.Timedelta(seconds=pipeline),
                         "created": received - pd.Timedelta(seconds=source)})  # fmt: skip


def test_recent_health_over_the_last_minutes():
    recent = times(range(0, 300, 2))  # 150 posts over 5 minutes
    health = data.recent_health(recent, written=500, windows=invented_windows(), first_received=T0 - pd.Timedelta(hours=1))
    assert health.period == "the last 5 minutes" and health.written == 500
    assert health.posts_per_second == pytest.approx(150 / 300, abs=0.01)  # newest at 298 s, so the window starts at -2 s
    assert (health.pipeline_median_s, health.pipeline_p95_s, health.source_median_s) == (4.0, 4.0, 2.0)
    assert health.last_received == T0 + pd.Timedelta(seconds=298)
    assert health.received is None and health.reconnects is None and health.endpoint_switches is None
    assert health.count_check["match"] is False  # 500 written against 14 in the windows


def test_recent_health_at_the_start_of_a_run_divides_by_the_time_covered():
    recent = times(range(0, 60))  # one post a second for the first minute
    health = data.recent_health(recent, written=60, windows=invented_windows(), first_received=T0)
    assert health.posts_per_second == pytest.approx(60 / 59, abs=0.01)


def test_recent_health_without_posts():
    health = data.recent_health(times([]), written=0, windows=invented_windows().iloc[:0], first_received=None)
    assert health.posts_per_second is None and health.pipeline_median_s is None
    assert health.count_check == {"predictions": 0, "all": 0, "topics": 0, "match": True}


def test_times_frame_reads_only_times_as_utc():
    documents = [{"received_at_ms": 1_800_000_000_000, "written_at": pd.Timestamp("2027-01-15 08:00:05").to_pydatetime(),
                  "created_at": None}]  # fmt: skip
    frame = data.times_frame(documents)
    assert list(frame.columns) == ["received", "written", "created"]
    assert frame.loc[0, "received"] == pd.Timestamp(1_800_000_000_000, unit="ms", tz="UTC")
    assert frame.loc[0, "written"] == pd.Timestamp("2027-01-15T08:00:05Z")
    assert pd.isna(frame.loc[0, "created"])


def test_out_of_domain_warning_numbers_come_from_the_shift_report():
    summary = data.domain_shift_summary(invented_shift())
    assert summary == {"n": 50, "skipped": 10, "macro_f1": 0.4, "macro_f1_ci95": [0.3, 0.5], "macro_f1_2015": 0.7,
                       "macro_f1_ci95_2015": [0.68, 0.72], "model_negative": 30, "human_negative": 10}  # fmt: skip
    text = out_of_domain_text(invented_shift())
    for part in ("50 hand-labelled", "0.400 (95% CI 0.300 to 0.500)", "0.700 on the 2015", "30 against 10",
                 "not the true share"):  # fmt: skip
        assert part in text
    assert "prediction, not a measured share" in out_of_domain_text(None)


# --- the page, on the recorded run in reports/ --------------------------------------------


@pytest.fixture
def recorded_page(monkeypatch):
    monkeypatch.setenv("SENTIMENT_SOURCE", "snapshot")
    monkeypatch.delenv("SENTIMENT_SNAPSHOT_DIR", raising=False)
    at = AppTest.from_file(str(config.BASE_DIR) + "/app.py", default_timeout=60)
    at.session_state["playing"] = False
    return at.run()


def plotly_values(values) -> np.ndarray:
    """Plotly 6 sends numeric arrays as {"dtype", "bdata": base64}; older versions as lists."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values)


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


def test_live_tab_shows_the_recorded_run(recorded_page):
    at = recorded_page
    assert not at.exception
    run = data.read_json(data.LIVE_REPORT_PATH)
    shift = data.read_json(data.DOMAIN_SHIFT_PATH)
    assert [tab.label for tab in at.tabs] == ["Airline tweets (replay)", "Bluesky live run"]
    banners = [banner.value for banner in at.tabs[1].info]
    assert banners == [data.recording_label(run)] and LIVE_MONGO_BANNER not in banners

    assert metric(at, "Posts received").value == f"{run['posts']['received_by_producer']:,}"
    assert metric(at, "Posts written").value == f"{run['posts']['written']:,}"
    assert metric(at, "Posts per second").value == f"{run['rate']['after_warmup']['mean_posts_per_second']:.1f}"
    pipeline = run["latency_seconds"]["after_warmup"]["pipeline"]
    assert metric(at, "Pipeline latency, median").value == f"{pipeline['median_s']:.1f} s"
    assert metric(at, "Count check").value == ("match" if run["count_check"]["match"] else "mismatch")
    assert metric(at, "Endpoint switches").value == str(run["connection"]["endpoint_switches"])
    assert [w.value for w in at.tabs[1].warning] == [out_of_domain_text(shift)]

    topics = next(d.value for d in at.dataframe if d.value.columns[0] == "Topic")
    assert topics["Posts"].sum() == run["count_check"]["windows_total_topic_rows"]
    charts = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    posts_chart = next(c for c in charts if c["data"][0]["type"] == "bar")
    assert plotly_values(posts_chart["data"][0]["y"]).sum() == run["count_check"]["windows_total_all_rows"]


def test_live_tab_never_shows_post_text(recorded_page):
    shown = json.dumps([d.value.to_dict() for d in recorded_page.dataframe], default=str)
    assert "did:" not in shown and "bsky.app" not in shown
    live_tables = [d.value for d in recorded_page.dataframe if d.value.columns[0] in ("Topic", "Minute (UTC)")]
    assert len(live_tables) == 2 and all("text" not in t.columns and "Tweet" not in t.columns for t in live_tables)
