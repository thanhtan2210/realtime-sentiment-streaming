"""Streamlit dashboard: negative share per airline, live from MongoDB or replayed from a snapshot.

Run from the repo root with `streamlit run app.py`.

SENTIMENT_SOURCE picks the data: `mongo`, `snapshot`, or `auto` (the default:
MongoDB if it answers within 2 seconds, otherwise the snapshot). The page
always says which one it is showing.
"""
import json
import os
import time
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

from src.common import config
from src.dashboard import data
from src.ml.data import load_config
from src.stream.settings import stream_settings

LIVE_BANNER = "Live from MongoDB"
REPLAY_BANNER = "Replay of a recorded run on {date}, not live Twitter data"
MODEL_NOTE = "Measured offline on the same two days that this stream replays."

# One fixed colour per airline (assigned in alphabetical order), so filtering never
# repaints a line. These six pass lightness, colour-blind separation and 3:1 contrast
# checks on both Streamlit's light and dark backgrounds.
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
REFERENCE_LINE_COLOR = "#8a8f98"


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


def render(view: data.View, all_airlines: list[str], window: pd.Timedelta, dashboard_cfg: dict) -> None:
    """The part of the page that is the same for live and replayed data."""
    if view.now is None:
        st.metric("Tweets processed", view.processed)
        st.write("Waiting for the stream processor to write its first predictions.")
        return

    selected = st.multiselect("Airlines", all_airlines, default=all_airlines, key="airlines")
    threshold = dashboard_cfg["negative_share_threshold"]

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
        figure = px.line(
            chart_rows,
            x="window_start",
            y="negative_share",
            color="airline",
            markers=True,
            hover_data={"total": True, "negative": True},
            labels={"window_start": "Window start (UTC)", "negative_share": "Negative share",
                    "airline": "Airline", "total": "Tweets", "negative": "Negative"},
            category_orders={"airline": all_airlines},
            color_discrete_map=dict(zip(all_airlines, SERIES_COLORS)),
        )  # fmt: skip
        figure.update_traces(line_width=2, marker_size=7)
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=REFERENCE_LINE_COLOR,
            line_width=1,
            annotation_text=f"threshold {threshold:.0%}",
            annotation_position="top left",
        )
        figure.update_yaxes(range=[0, 1.05], tickformat=".0%")
        figure.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=380, hovermode="x unified")
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
    st.caption("One point is one airline in one window. Windows with few tweets swing widely; hover for the count.")

    table_column, tweets_column = st.columns([2, 3])
    with table_column:
        st.subheader("Window in progress, by airline")
        st.caption(window_label(view.now.floor(window), window))
        table = data.current_window_table(view.windows, view.now, window, threshold, selected)
        st.dataframe(
            pd.DataFrame(
                {
                    "Airline": table["airline"],
                    "Tweets": table["total"],
                    "Negative": table["negative"],
                    "Negative share": table["negative_share"].map(percent),
                    "Above threshold": table["above_threshold"].map({True: "yes", False: "no"}),
                }
            ),
            hide_index=True,
            use_container_width=True,
            key="latest_window_table",
        )
        st.caption(f"Above threshold: negative share over {threshold:.0%} (config/pipeline.yaml).")
    with tweets_column:
        st.subheader(f"Latest {dashboard_cfg['latest_tweets']} tweets")
        latest = view.latest[view.latest["airline"].isin(selected)]
        st.dataframe(
            pd.DataFrame(
                {
                    "Created (UTC)": latest["created_at"].dt.strftime("%Y-%m-%d %H:%M"),
                    "Airline": latest["airline"],
                    "Predicted": latest["sentiment"],
                    "Tweet": latest["text"],
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


def main() -> None:
    st.set_page_config(page_title="Airline tweet sentiment", layout="wide")
    cfg = load_config()
    dashboard_cfg = cfg["dashboard"]
    settings = stream_settings(cfg)
    window = pd.Timedelta(settings["window"])
    refresh = dashboard_cfg["refresh_seconds"]
    state = st.session_state

    st.title("Airline tweet sentiment")
    st.write("Right now, are each airline's customers talking positively or negatively?")

    if "source" not in state:
        try:
            state["source"] = data.choose_source(
                os.environ.get("SENTIMENT_SOURCE", "auto"), settings["mongodb"]["uri"]
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

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
