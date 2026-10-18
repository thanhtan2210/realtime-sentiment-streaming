"""Streamlit dashboard, one tab per mode of the pipeline.

Run from the repo root with `streamlit run app.py`.

Airline tweets (replay): negative share per airline, live from MongoDB or
replayed from a snapshot. Bluesky live run: pipeline health, posts and the
model's negative share per minute, and the topic counts, from the recorded
run in reports/ or live from MongoDB; never the text of a live post.

SENTIMENT_SOURCE picks the data: `mongo`, `snapshot` (the recorded run in the
live tab), or `auto` (the default: MongoDB if it answers within 2 seconds,
otherwise the snapshot; the live tab also needs live windows in MongoDB).
Each tab always says which source it is showing.
"""
import html
import json
import os
import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

from src.common import config
from src.dashboard import data
from src.ml.data import load_config
from src.stream.settings import mode_settings, stream_settings

LIVE_BANNER = "Live from MongoDB"
REPLAY_BANNER = "Replay of a recorded run on {date}, not live Twitter data"
MODEL_NOTE = "Measured offline on the same two days that this stream replays."

# One fixed colour per airline (assigned in alphabetical order), so filtering never
# repaints a line. These six pass lightness, colour-blind separation and 3:1 contrast
# checks on both Streamlit's light and dark backgrounds.
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
REFERENCE_LINE_COLOR = "#8a8f98"
FADED_OPACITY = 0.4  # windows with too few tweets
TOO_FEW = "too few tweets"


@st.cache_data
def cached_snapshot(directory: str) -> data.Snapshot:
    return data.load_snapshot(directory)


@st.cache_resource
def mongo_database(uri: str, name: str):
    return MongoClient(uri, serverSelectionTimeoutMS=2000)[name]


@st.cache_data
def model_metrics() -> dict:
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def percent(share: float | None) -> str:
    return "n/a" if share is None else f"{share:.0%}"


def window_label(start: pd.Timestamp, window: pd.Timedelta) -> str:
    return f"{start:%Y-%m-%d %H:%M} to {start + window:%H:%M} UTC"


def span_label(window: pd.Timedelta) -> str:
    """ "hour" for a one-hour window, otherwise something like "30 minutes"."""
    minutes = int(window.total_seconds() // 60)
    return "hour" if minutes == 60 else f"{minutes} minutes"


def share_metric(column, label: str, kpi: dict, span: str) -> None:
    delta = None
    if kpi["share"] is not None and kpi["previous_share"] is not None:
        delta = f"{(kpi['share'] - kpi['previous_share']) * 100:+.0f} pp vs the {span} before"
    # A rising negative share is bad news, so an increase is drawn in red.
    column.metric(label, percent(kpi["share"]), delta, delta_color="inverse")
    column.caption(f"{kpi['tweets']} tweets in the last {span}, {kpi['previous_tweets']} in the {span} before")


def negative_share_figure(
    rows: pd.DataFrame, all_airlines: list[str], threshold: float, min_tweets: int
) -> go.Figure:
    """Negative share per window, one colour per airline.

    A window with fewer than `min_tweets` tweets is a faded marker, and the
    line into and out of it is faded and dotted; the rest is solid.
    """
    figure = go.Figure()
    hover = "%{y:.0%} of %{customdata[0]} tweets (%{customdata[1]} negative)"
    for airline, color in zip(all_airlines, SERIES_COLORS):
        series = rows[rows["airline"] == airline].sort_values("window_start")
        if series.empty:
            continue
        x, share = series["window_start"], series["negative_share"]
        counts = series[["total", "negative"]].to_numpy()
        enough = (series["total"] >= min_tweets).to_numpy()
        # A point belongs to the dotted line if it or one of its neighbours has too few tweets.
        thin = ~enough
        thin[:-1] |= ~enough[1:]
        thin[1:] |= ~enough[:-1]
        figure.add_trace(
            go.Scatter(
                x=x, y=share.where(enough), customdata=counts, name=airline, legendgroup=airline,
                mode="lines+markers", connectgaps=False,
                line={"color": color, "width": 2}, marker={"color": color, "size": 7},
                hovertemplate=hover,
            )
        )  # fmt: skip
        figure.add_trace(
            go.Scatter(
                x=x, y=share.where(thin), name=airline, legendgroup=airline, showlegend=False,
                mode="lines", connectgaps=False, opacity=FADED_OPACITY,
                line={"color": color, "width": 2, "dash": "dot"}, hoverinfo="skip",
            )
        )  # fmt: skip
        figure.add_trace(
            go.Scatter(
                x=x, y=share.where(~enough), customdata=counts, name=airline, legendgroup=airline,
                showlegend=False, mode="markers", opacity=FADED_OPACITY,
                marker={"color": color, "size": 7},
                hovertemplate=hover + f", {TOO_FEW}",
            )
        )  # fmt: skip
    figure.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=REFERENCE_LINE_COLOR,
        line_width=1,
        annotation_text=f"threshold {threshold:.0%}",
        annotation_position="top left",
    )
    figure.update_xaxes(title_text="Window start (UTC)")
    figure.update_yaxes(title_text="Negative share", range=[0, 1.05], tickformat=".0%")
    figure.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=380,
        hovermode="x unified",
        legend_title_text="Airline",
        legend_tracegroupgap=0,
    )
    return figure


def interval_text(low: float, high: float) -> str:
    return f"{low:.0%} to {high:.0%}"


def flag_text(enough_tweets: bool, above_threshold: bool) -> str:
    if not enough_tweets:
        return TOO_FEW
    return "yes" if above_threshold else "no"


def render(view: data.View, all_airlines: list[str], window: pd.Timedelta, dashboard_cfg: dict) -> None:
    """The part of the page that is the same for live and replayed data."""
    if view.now is None:
        st.metric("Tweets processed", view.processed)
        st.write("Waiting for the stream processor to write its first predictions.")
        return

    selected = st.multiselect("Airlines", all_airlines, default=all_airlines, key="airlines")
    threshold = dashboard_cfg["negative_share_threshold"]
    min_tweets = dashboard_cfg["min_tweets"]

    span = span_label(window)
    everyone = data.last_hour_kpi(view.recent, view.now, window)
    chosen = data.last_hour_kpi(view.recent, view.now, window, selected)
    processed, overall, selection = st.columns(3)
    processed.metric("Tweets processed", f"{view.processed:,}")
    processed.caption(f"Up to {view.now:%Y-%m-%d %H:%M} UTC (tweet time)")
    share_metric(overall, f"Negative share in the last {span}, all airlines", everyone, span)
    share_metric(selection, f"Negative share in the last {span}, selected airlines", chosen, span)

    st.subheader("Negative share per airline, by window")
    chart_rows = view.windows[view.windows["airline"].isin(selected)]
    if chart_rows.empty:
        st.write("No tweets for the selected airlines yet.")
    else:
        figure = negative_share_figure(chart_rows, all_airlines, threshold, min_tweets)
        st.plotly_chart(figure, use_container_width=True, key="negative_share_chart")
        with st.expander("Chart data as a table"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Window start (UTC)": chart_rows["window_start"].dt.strftime("%Y-%m-%d %H:%M"),
                        "Airline": chart_rows["airline"],
                        "Tweets": chart_rows["total"],
                        "Negative": chart_rows["negative"],
                        "Negative share": chart_rows["negative_share"].map(percent),
                    }
                ),
                hide_index=True,
                use_container_width=True,
                key="chart_table",
            )
    st.caption(
        f"One point is one airline in one window. Windows with fewer than {min_tweets} tweets swing "
        "widely: they are drawn faded, with a dotted line. Hover for the count."
    )

    table_column, tweets_column = st.columns([2, 3])
    with table_column:
        st.subheader("Window in progress, by airline")
        st.caption(window_label(view.now.floor(window), window))
        table = data.current_window_table(view.windows, view.now, window, threshold, min_tweets, selected)
        st.dataframe(
            pd.DataFrame(
                {
                    "Airline": table["airline"],
                    "Tweets": table["total"],
                    "Negative": table["negative"],
                    "Negative share": table["negative_share"].map(percent),
                    "95% CI": list(map(interval_text, table["ci_low"], table["ci_high"])),
                    "Above threshold": list(map(flag_text, table["enough_tweets"], table["above_threshold"])),
                }
            ),
            hide_index=True,
            use_container_width=True,
            key="latest_window_table",
        )
        st.caption(
            f"Above threshold: at least {min_tweets} tweets and a negative share over {threshold:.0%} "
            "(config/pipeline.yaml). 95% CI: Wilson interval for the negative share."
        )
    with tweets_column:
        st.subheader(f"Latest {dashboard_cfg['latest_tweets']} tweets")
        latest = view.latest[view.latest["airline"].isin(selected)]
        st.dataframe(
            pd.DataFrame(
                {
                    "Created (UTC)": latest["created_at"].dt.strftime("%Y-%m-%d %H:%M"),
                    "Airline": latest["airline"],
                    "Predicted": latest["sentiment"],
                    # Display only: the stored text and the model's input keep entities like &amp;.
                    "Tweet": latest["text"].map(html.unescape),
                }
            ),
            hide_index=True,
            use_container_width=True,
            key="latest_tweets_table",
        )


def advance_clock(state, end: pd.Timestamp) -> None:
    """Move the simulated clock by the real time since the last refresh, times the speed."""
    tick = time.monotonic()
    if state["playing"]:
        elapsed = timedelta(seconds=state["speed"] * (tick - state["last_tick"]))
        state["clock"] = min(end, state["clock"] + elapsed)
    state["last_tick"] = tick


def replay_panel(snapshot: data.Snapshot, window: pd.Timedelta, dashboard_cfg: dict) -> None:
    state = st.session_state
    start, end = snapshot.predictions["created_at"].min(), snapshot.predictions["created_at"].max()
    advance_clock(state, end)
    if state["playing"] and state["clock"] >= end:
        state["playing"] = False
        st.rerun()  # a full rerun drops the refresh timer

    play, reset, speed, clock = st.columns([1, 1, 2, 5])
    if play.button("Pause" if state["playing"] else "Play", use_container_width=True):
        if state["clock"] >= end:
            state["clock"] = start + window
        state["playing"] = not state["playing"]
        st.rerun()
    if reset.button("Reset", use_container_width=True):
        state["clock"], state["playing"] = start + window, False
        st.rerun()
    speed.selectbox(
        "Speed",
        dashboard_cfg["replay_speeds"],
        key="speed",
        format_func=lambda value: f"{value}x real time",
        label_visibility="collapsed",
    )
    clock.progress(
        float((state["clock"] - start) / (end - start)),
        text=f"Simulated clock: {state['clock']:%Y-%m-%d %H:%M:%S} UTC",
    )

    view = data.replay_view(snapshot, state["clock"], window, dashboard_cfg["latest_tweets"])
    render(view, sorted(snapshot.predictions["airline"].unique()), window, dashboard_cfg)


def live_panel(mongodb_cfg: dict, window: pd.Timedelta, dashboard_cfg: dict) -> None:
    database = mongo_database(mongodb_cfg["uri"], mongodb_cfg["database"])
    view = data.live_view(database, mongodb_cfg, window, dashboard_cfg["latest_tweets"])
    render(view, sorted(view.windows["airline"].unique()), window, dashboard_cfg)


def about_the_model() -> None:
    metrics = model_metrics()
    test, split = metrics["test"], metrics["split"]
    model, baseline = test["model"], test["baseline"]
    with st.container(border=True):
        st.subheader("About the model")
        st.write(
            "TF-IDF + logistic regression, three classes. Scores on the held-out test tweets "
            f"({split['test']['rows']:,} tweets, the last two days of the dataset):"
        )
        columns = st.columns(1 + len(test["labels"]))
        low, high = model["macro_f1_ci95"]
        columns[0].metric("Macro-F1", f"{model['macro_f1']:.3f}")
        columns[0].caption(
            f"95% CI {low:.3f} to {high:.3f}. Always predicting "
            f"'{test['baseline_predicts']}' scores {baseline['macro_f1']:.3f}."
        )
        for column, label in zip(columns[1:], test["labels"]):
            column.metric(f"F1, {label}", f"{model['f1_per_class'][label]:.3f}")
        st.caption(f"{MODEL_NOTE} Source: models/metrics.json, written by `python -m src.ml.train`.")


# --- the Bluesky live run ---------------------------------------------------------------

LIVE_MONGO_BANNER = "Live from MongoDB: live_windows and the times in live_predictions (no post text is read)"
LIVE_MINUTES = 5  # the MongoDB source reports pipeline health over this many recent minutes
POSTS_COLOR = SERIES_COLORS[0]
NOT_IN_MONGO = "only in the producer's log, so only for the recorded run"


@st.cache_data
def cached_live_recording() -> tuple[dict, pd.DataFrame]:
    return data.read_json(data.LIVE_REPORT_PATH), data.load_live_windows()


@st.cache_data
def cached_domain_shift() -> dict | None:
    return data.read_json(data.DOMAIN_SHIFT_PATH)


def count_text(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def seconds_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def health_panel(health: data.LiveHealth) -> None:
    st.subheader("Pipeline health")
    when = f"Rate and latencies over {health.period}; post counts over the whole run."
    if health.last_received is not None:
        when += f" Last post received at {health.last_received:%Y-%m-%d %H:%M:%S} UTC."
    st.caption(when)
    check = health.count_check
    first = st.columns(4)
    first[0].metric("Posts received", count_text(health.received))
    first[0].caption("Read by the producer" if health.received is not None else f"Posts received: {NOT_IN_MONGO}")
    first[1].metric("Posts written", count_text(health.written))
    first[1].caption("To MongoDB (live_predictions)")
    first[2].metric("Posts per second", "n/a" if health.posts_per_second is None else f"{health.posts_per_second:.1f}")
    first[3].metric("Count check", "match" if check["match"] else "mismatch")
    first[3].caption(
        f"{check['predictions']:,} posts written, {check['all']:,} in the `all` windows, "
        f"{check['topics']:,} across the topics"
    )
    second = st.columns(4)
    second[0].metric("Pipeline latency, median", seconds_text(health.pipeline_median_s))
    second[0].caption(f"p95 {seconds_text(health.pipeline_p95_s)}. Producer to Kafka to Spark to MongoDB.")
    second[1].metric("Source latency, median", seconds_text(health.source_median_s))
    second[1].caption("From the post's createdAt, set by the poster's app, to the producer.")
    second[2].metric("Reconnects", count_text(health.reconnects))
    second[3].metric("Endpoint switches", count_text(health.endpoint_switches))
    if health.reconnects is None:
        second[2].caption(f"Reconnects and endpoint switches: {NOT_IN_MONGO}")


def posts_per_minute_figure(minutes: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=minutes["window_start"], y=minutes["total"], marker={"color": POSTS_COLOR},
            name="Posts", hovertemplate="%{x|%H:%M} UTC: %{y:,} posts<extra></extra>",
        )
    )  # fmt: skip
    figure.update_xaxes(title_text="Minute (UTC)")
    figure.update_yaxes(title_text="Posts", rangemode="tozero")
    figure.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=320, bargap=0.15)
    return figure


def live_share_figure(minutes: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=minutes["window_start"], y=minutes["negative_share"], customdata=minutes[["total", "negative"]],
            mode="lines+markers", line={"color": POSTS_COLOR, "width": 2}, marker={"color": POSTS_COLOR, "size": 8},
            name="Predicted negative share",
            hovertemplate="%{x|%H:%M} UTC: %{y:.0%} predicted negative (%{customdata[1]:,} of %{customdata[0]:,})"
            "<extra></extra>",
        )
    )  # fmt: skip
    figure.update_xaxes(title_text="Minute (UTC)")
    figure.update_yaxes(title_text="Predicted negative share", range=[0, 1.05], tickformat=".0%")
    figure.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=320)
    return figure


def out_of_domain_text(shift: dict | None) -> str:
    if shift is None:
        return (
            "**Out of domain.** The model was trained on 2015 airline tweets and has not been scored on "
            "Bluesky posts. This line is the model's prediction, not a measured share."
        )
    s = data.domain_shift_summary(shift)
    low, high = s["macro_f1_ci95"]
    return (
        f"**Out of domain.** The model was trained on 2015 airline tweets. On {s['n']} hand-labelled "
        f"Bluesky posts its macro-F1 is {s['macro_f1']:.3f} (95% CI {low:.3f} to {high:.3f}), against "
        f"{s['macro_f1_2015']:.3f} on the 2015 test tweets, and it labels more posts negative than the "
        f"annotator did ({s['model_negative']} against {s['human_negative']}). This line is the model's "
        "prediction, not the true share of negative posts. Source: reports/live_domain_shift.json, one annotator."
    )


def live_charts(windows: pd.DataFrame, still_running: bool) -> None:
    minutes = data.per_minute(windows)
    if minutes.empty:
        st.write("No live windows yet.")
        return
    edge = "The current minute is still filling." if still_running else "The first and last minutes are partial."
    posts_column, share_column = st.columns(2)
    with posts_column:
        st.subheader("Posts per minute, all topics")
        st.plotly_chart(posts_per_minute_figure(minutes), use_container_width=True, key="live_posts_chart")
        st.caption(f"1-minute event-time windows (the `all` row). {edge}")
    with share_column:
        st.subheader("Negative share per minute, predicted")
        st.warning(out_of_domain_text(cached_domain_shift()))
        st.plotly_chart(live_share_figure(minutes), use_container_width=True, key="live_share_chart")
    with st.expander("Chart data as a table"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Minute (UTC)": minutes["window_start"].dt.strftime("%Y-%m-%d %H:%M"),
                    "Posts": minutes["total"],
                    "Predicted negative": minutes["negative"],
                    "Predicted negative share": minutes["negative_share"].map(percent),
                }
            ),
            hide_index=True,
            use_container_width=True,
            key="live_minutes_table",
        )


def topics_table(windows: pd.DataFrame) -> None:
    st.subheader("Topics")
    counts = data.topic_counts(windows)
    st.dataframe(
        pd.DataFrame({"Topic": counts["topic"], "Posts": counts["posts"]}),
        hide_index=True,
        key="live_topics_table",
    )
    st.caption(
        "general: neither of the others; air_travel: talks about flying; airline:<name>: names an airline "
        "(config/live_sources.yaml). Counts only: these posts are too rare on Bluesky to chart."
    )


def render_live(health: data.LiveHealth, windows: pd.DataFrame, still_running: bool) -> None:
    health_panel(health)
    live_charts(windows, still_running)
    topics_table(windows)


def live_mongo_panel(mongodb_cfg: dict) -> None:
    database = mongo_database(mongodb_cfg["uri"], mongodb_cfg["database"])
    recent, written, windows, first = data.live_mongo_data(database, mongodb_cfg, LIVE_MINUTES)
    render_live(data.recent_health(recent, written, windows, first, LIVE_MINUTES), windows, still_running=True)


def live_recorded_panel() -> None:
    run, windows = cached_live_recording()
    render_live(data.recorded_health(run), windows, still_running=False)
    st.caption(
        "Source: reports/live_run.json and reports/live_snapshot/windows.parquet, written by "
        "`python -m scripts.live_report` in .github/workflows/live-run.yml."
    )


def live_tab(live_mongodb: dict, refresh: int) -> None:
    state = st.session_state
    if "live_source" not in state:
        state["live_source"] = data.choose_live_source(
            os.environ.get("SENTIMENT_SOURCE", "auto"),
            lambda: data.mongo_has_live_windows(
                live_mongodb["uri"], live_mongodb["database"], live_mongodb["windows_collection"]
            ),
        )
    if state["live_source"] == "mongo":
        st.info(LIVE_MONGO_BANNER)
        st.fragment(run_every=refresh)(live_mongo_panel)(live_mongodb)
    else:
        run, _ = cached_live_recording()
        st.info(data.recording_label(run))
        live_recorded_panel()


def main() -> None:
    st.set_page_config(page_title="Post sentiment stream", layout="wide")
    cfg = load_config()
    dashboard_cfg = cfg["dashboard"]
    settings = stream_settings(cfg)
    window = pd.Timedelta(settings["window"])
    refresh = dashboard_cfg["refresh_seconds"]
    state = st.session_state

    st.title("Post sentiment stream")
    st.write(
        "Every incoming post is labelled negative, neutral or positive and counted per time window: "
        "accuracy is checked on replayed airline tweets, live behaviour on the Bluesky firehose."
    )

    if "source" not in state:
        try:
            state["source"] = data.choose_source(
                os.environ.get("SENTIMENT_SOURCE", "auto"), settings["mongodb"]["uri"]
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    replay_tab, bluesky_tab = st.tabs(["Airline tweets (replay)", "Bluesky live run"])
    with replay_tab:
        if state["source"] == "mongo":
            st.info(LIVE_BANNER)
            st.fragment(run_every=refresh)(live_panel)(settings["mongodb"], window, dashboard_cfg)
        else:
            snapshot = cached_snapshot(data.snapshot_dir(dashboard_cfg))
            st.info(REPLAY_BANNER.format(date=snapshot.meta["recorded_at_utc"][:10]))
            state.setdefault("clock", snapshot.predictions["created_at"].min() + window)
            state.setdefault("playing", True)
            state.setdefault("speed", dashboard_cfg["replay_speeds"][0])
            state.setdefault("last_tick", time.monotonic())
            timer = refresh if state["playing"] else None
            st.fragment(run_every=timer)(replay_panel)(snapshot, window, dashboard_cfg)
        about_the_model()
    with bluesky_tab:
        live_tab(mode_settings(cfg, "live")["mongodb"], refresh)
