"""The dashboard in replay mode, on a tiny hand-made snapshot (no MongoDB, no Docker)."""
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.common import config
from src.dashboard import data
from src.dashboard.app import LIVE_BANNER, MODEL_NOTE, negative_share_figure

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
                    "text": f"@{airline} delays &amp; refunds, tweet number {i}",
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


@pytest.mark.parametrize(
    "negative, tweets, low, high",
    [
        (0, 10, 0.0, 0.2775),
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
        (1, 1, 0.2065, 1.0),
        (45, 50, 0.7864, 0.9565),
    ],
)
def test_wilson_interval_matches_known_values(negative, tweets, low, high):
    assert data.wilson_interval(negative, tweets) == pytest.approx((low, high), abs=5e-5)


def test_wilson_interval_needs_at_least_one_tweet():
    with pytest.raises(ValueError):
        data.wilson_interval(0, 0)


def test_current_window_table_flags_only_windows_with_enough_tweets():
    windows = data.window_counts(fixture_predictions(), HOUR)
    # Hour 9: American 1 of 1 negative, United 4 of 5, Delta 1 of 4.
    rows = data.current_window_table(windows, END_OF_HOUR_9, HOUR, threshold=0.75, min_tweets=4)
    by_airline = rows.set_index("airline")
    assert by_airline["enough_tweets"].to_dict() == {"American": False, "United": True, "Delta": True}
    # American is at 100% but on a single tweet: not flagged.
    assert by_airline["above_threshold"].to_dict() == {"American": False, "United": True, "Delta": False}
    assert list(rows["negative_share"]) == sorted(rows["negative_share"], reverse=True)
    united = by_airline.loc["United"]
    assert (united["ci_low"], united["ci_high"]) == pytest.approx(data.wilson_interval(4, 5))

    # With the bar above every count, nothing is flagged, whatever the share.
    strict = data.current_window_table(windows, END_OF_HOUR_9, HOUR, threshold=0.75, min_tweets=20)
    assert not strict["enough_tweets"].any() and not strict["above_threshold"].any()


def test_chart_fades_windows_with_too_few_tweets():
    windows = data.window_counts(fixture_predictions(), HOUR)
    figure = negative_share_figure(windows, ["American", "Delta", "United"], threshold=0.75, min_tweets=4)
    united = [trace for trace in figure.data if trace.name == "United"]
    solid, dotted, faded = united
    # United has 4, 5 and 2 tweets in hours 8, 9 and 10: the last window is below the bar.
    assert [y == y for y in solid.y] == [True, True, False]  # NaN where there are too few tweets
    assert [y == y for y in faded.y] == [False, False, True]
    assert (faded.mode, faded.opacity) == ("markers", 0.4)
    # The dotted line runs from the last solid point into the faded one.
    assert [y == y for y in dotted.y] == [False, True, True]
    assert (dotted.line.dash, dotted.opacity, dotted.hoverinfo) == ("dot", 0.4, "skip")
    assert solid.customdata.tolist() == [[4, 3], [5, 4], [2, 1]]  # tweets, negative: shown on hover
    assert "too few tweets" in faded.hovertemplate


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
    # Every window of the fixture has fewer than the configured 20 tweets, so none is flagged.
    latest = table(at, "Airline")
    assert latest.values.tolist() == [
        ["American", 1, 1, "100%", "21% to 100%", "too few tweets"],
        ["United", 5, 4, "80%", "38% to 96%", "too few tweets"],
        ["Delta", 4, 1, "25%", "5% to 70%", "too few tweets"],
    ]
    assert any(
        "Above threshold: at least 20 tweets and a negative share over 75%" in caption.value
        for caption in at.caption
    )


def test_tweet_text_is_unescaped_for_display_only(snapshot_dir):
    at = open_app(END_OF_HOUR_9)
    shown = table(at, "Created (UTC)")["Tweet"]
    assert shown.str.contains("delays & refunds").all()
    assert not shown.str.contains("&amp;").any()
    # the data behind the page still holds the text exactly as stored
    stored = data.load_snapshot(str(snapshot_dir)).predictions["text"]
    assert stored.str.contains("&amp;").all()


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
