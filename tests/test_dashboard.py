"""The dashboard in replay mode, on a tiny hand-made snapshot (no MongoDB, no Docker)."""
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.common import config
from src.dashboard import data
from src.dashboard.app import LIVE_BANNER, MODEL_NOTE

HOUR = pd.Timedelta("1 hour")
# (hour, airline, sentiment, how many). Tweets are spread over the minutes of their hour.
FIXTURE = [
    (8, "United", "negative", 3), (8, "United", "positive", 1),
    (8, "Delta", "neutral", 1), (8, "Delta", "positive", 1),
    (9, "United", "negative", 4), (9, "United", "neutral", 1),
    (9, "Delta", "negative", 1), (9, "Delta", "positive", 3),
    (9, "American", "negative", 1),
    (10, "United", "negative", 1), (10, "United", "positive", 1),
    (10, "Delta", "neutral", 2),
]  # fmt: skip
END_OF_HOUR_9 = pd.Timestamp("2015-02-23T09:59:59Z")


def fixture_predictions() -> pd.DataFrame:
    rows = []
    for hour, airline, sentiment, count in FIXTURE:
        for _ in range(count):
            i = len(rows)
            rows.append(
                {
                    "tweet_id": str(5700 + i),
                    "airline": airline,
                    "text": f"@{airline} tweet number {i}",
                    "created_at": pd.Timestamp(f"2015-02-23T{hour:02d}:00:00Z") + pd.Timedelta(minutes=1 + i),
                    "sentiment": sentiment,
                    "produced_at_ms": 1_700_000_000_000 + i,
                    "written_at": pd.Timestamp("2026-01-02T03:04:05Z"),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    predictions = fixture_predictions()
    predictions.to_parquet(tmp_path / "predictions.parquet", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({"recorded_at_utc": "2026-01-02T03:04:05Z"}), encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_SOURCE", "snapshot")
    monkeypatch.setenv("SENTIMENT_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def open_app(clock: pd.Timestamp) -> AppTest:
    """The app paused with its simulated clock at `clock`."""
    at = AppTest.from_file(str(config.BASE_DIR) + "/app.py", default_timeout=60)
    at.session_state["clock"] = clock
    at.session_state["playing"] = False
    return at.run()


def metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def table(at: AppTest, first_column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if d.value.columns[0] == first_column)


# --- the calculations -------------------------------------------------------------------


def test_window_counts_match_the_fixture():
    windows = data.window_counts(fixture_predictions(), HOUR)
    united_9 = windows[(windows["airline"] == "United") & (windows["window_start"] == "2015-02-23T09:00:00Z")]
    assert united_9[["negative", "neutral", "positive", "total"]].values.tolist() == [[4, 1, 0, 5]]
    assert united_9["negative_share"].iloc[0] == pytest.approx(0.8)
    assert int(windows["total"].sum()) == sum(count for *_, count in FIXTURE)


def test_replay_view_only_contains_tweets_up_to_the_clock():
    snapshot = data.Snapshot(fixture_predictions(), {})
    view = data.replay_view(snapshot, END_OF_HOUR_9, HOUR, latest=20)
    assert view.processed == 16  # 6 tweets in hour 8, 10 in hour 9
    assert view.latest["created_at"].max() <= END_OF_HOUR_9
    assert view.latest["created_at"].is_monotonic_decreasing
    assert view.windows["window_start"].max() == pd.Timestamp("2015-02-23T09:00:00Z")


def test_last_hour_kpi_compares_a_rolling_hour_with_the_hour_before():
    view = data.replay_view(data.Snapshot(fixture_predictions(), {}), END_OF_HOUR_9, HOUR, latest=20)
    everyone = data.last_hour_kpi(view.recent, END_OF_HOUR_9, HOUR)
    assert (everyone["share"], everyone["tweets"]) == (pytest.approx(0.6), 10)  # 6 of 10
    assert (everyone["previous_share"], everyone["previous_tweets"]) == (pytest.approx(0.5), 6)  # 3 of 6
    united = data.last_hour_kpi(view.recent, END_OF_HOUR_9, HOUR, ["United"])
    assert (united["share"], united["previous_share"]) == (pytest.approx(0.8), pytest.approx(0.75))
    nobody = data.last_hour_kpi(view.recent, END_OF_HOUR_9, HOUR, [])
    assert (nobody["share"], nobody["tweets"]) == (None, 0)


def test_last_hour_is_rolling_not_the_clock_hour_in_progress():
    # Two minutes into hour 10 the clock hour is still empty; the rolling hour holds all ten
    # tweets of hour 9, and the hour before it the four tweets sent after 08:02.
    clock = pd.Timestamp("2015-02-23T10:02:00Z")
    view = data.replay_view(data.Snapshot(fixture_predictions(), {}), clock, HOUR, latest=20)
    in_progress = view.windows[view.windows["window_start"] == clock.floor(HOUR)]
    assert int(in_progress["total"].sum()) == 0
    kpi = data.last_hour_kpi(view.recent, clock, HOUR)
    assert kpi["tweets"] == 10 and kpi["previous_tweets"] == 4
    assert view.recent["created_at"].min() > clock - 2 * HOUR


def test_current_window_table_flags_shares_above_the_threshold():
    windows = data.window_counts(fixture_predictions(), HOUR)
    rows = data.current_window_table(windows, END_OF_HOUR_9, HOUR, threshold=0.75)
    flagged = dict(zip(rows["airline"], rows["above_threshold"]))
    assert flagged == {"American": True, "United": True, "Delta": False}  # 100%, 80%, 25%
    assert list(rows["negative_share"]) == sorted(rows["negative_share"], reverse=True)


def test_source_choice(monkeypatch):
    assert data.choose_source("snapshot", "mongodb://unused") == "snapshot"
    assert data.choose_source("mongo", "mongodb://unused") == "mongo"
    with pytest.raises(ValueError):
        data.choose_source("kafka", "mongodb://unused")
    monkeypatch.setattr(data, "mongo_reachable", lambda uri: False)
    assert data.choose_source("auto", "mongodb://unused") == "snapshot"
    monkeypatch.setattr(data, "mongo_reachable", lambda uri: True)
    assert data.choose_source("auto", "mongodb://unused") == "mongo"


def test_auto_falls_back_to_the_snapshot_when_nothing_listens():
    assert data.mongo_reachable("mongodb://127.0.0.1:9", timeout_ms=300) is False


# --- the page ---------------------------------------------------------------------------


def test_replay_mode_says_it_is_a_replay(snapshot_dir):
    at = open_app(END_OF_HOUR_9)
    assert not at.exception
    banners = [banner.value for banner in at.info]
    assert banners == ["Replay of a recorded run on 2026-01-02, not live Twitter data"]
    assert LIVE_BANNER not in banners
    assert [button.label for button in at.button] == ["Play", "Reset"]


def test_page_at_clock_t_shows_only_data_up_to_t(snapshot_dir):
    at = open_app(END_OF_HOUR_9)
    assert not at.exception
    assert metric(at, "Tweets processed").value == "16"
    tweets = table(at, "Created (UTC)")
    assert len(tweets) == 16
    assert tweets["Created (UTC)"].max() <= "2015-02-23 09:59"
    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    points = [x for trace in chart["data"] if trace.get("mode") for x in trace["x"]]
    assert points and max(points) < "2015-02-23T10"

    earlier = open_app(pd.Timestamp("2015-02-23T08:59:59Z"))
    assert metric(earlier, "Tweets processed").value == "6"


def test_kpis_are_computed_from_the_fixture(snapshot_dir):
    at = open_app(END_OF_HOUR_9)
    overall = metric(at, "Negative share in the last hour, all airlines")
    assert (overall.value, overall.delta) == ("60%", "+10 pp vs the hour before")  # 6/10 after 3/6
    assert metric(at, "Negative share in the last hour, selected airlines").value == "60%"  # all selected
    assert any("10 tweets in the last hour, 6 in the hour before" in c.value for c in at.caption)
    latest = table(at, "Airline")
    assert latest.values.tolist() == [
        ["American", 1, 1, "100%", "yes"],
        ["United", 5, 4, "80%", "yes"],
        ["Delta", 4, 1, "25%", "no"],
    ]
    assert any("Above threshold: negative share over 75%" in caption.value for caption in at.caption)


def test_changing_the_airlines_does_not_raise(snapshot_dir):
    at = open_app(END_OF_HOUR_9)
    at.multiselect(key="airlines").set_value(["United"]).run()
    assert not at.exception
    selected = metric(at, "Negative share in the last hour, selected airlines")
    assert (selected.value, selected.delta) == ("80%", "+5 pp vs the hour before")  # 4/5 after 3/4
    assert metric(at, "Negative share in the last hour, all airlines").value == "60%"
    assert table(at, "Airline")["Airline"].tolist() == ["United"]
    assert set(table(at, "Created (UTC)")["Airline"]) == {"United"}

    at.multiselect(key="airlines").set_value([]).run()
    assert not at.exception
    assert metric(at, "Negative share in the last hour, selected airlines").value == "n/a"


def test_about_the_model_reads_metrics_json(snapshot_dir):
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        model = json.load(f)["test"]["model"]
    at = open_app(END_OF_HOUR_9)
    assert metric(at, "Macro-F1").value == f"{model['macro_f1']:.3f}"
    for label, score in model["f1_per_class"].items():
        assert metric(at, f"F1, {label}").value == f"{score:.3f}"
    assert any(MODEL_NOTE in caption.value for caption in at.caption)


# --- the committed snapshot --------------------------------------------------------------


def test_spark_windows_equal_the_replay_mode_aggregation():
    """Replay mode recounts windows in pandas; at the end of the run it must match Spark's sink."""
    directory = data.snapshot_dir({"snapshot_dir": "reports/stream_snapshot"}, env={})
    snapshot = data.load_snapshot(directory)
    assert snapshot.predictions["created_at"].is_monotonic_increasing
    recounted = data.window_counts(snapshot.predictions, HOUR)
    stored = pd.read_parquet(directory + "/windows.parquet")[data.WINDOW_COLUMNS]
    stored = stored.sort_values(["window_start", "airline"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(stored, recounted, check_dtype=False)
