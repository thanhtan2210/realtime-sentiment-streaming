"""Endpoint switching on source lag, and a clean stop on SIGTERM, against a fake Jetstream.

Time is simulated: each event moves a fake clock forward by half a second, so
minutes of lag pass in milliseconds and the results do not depend on the machine.
Every DID, rkey and text is invented.
"""
import asyncio
import json
import re
import signal
from datetime import datetime, timezone

import pytest

from src.stream.sources.bluesky import CursorStore, DeliveryTracker, KafkaSink, MinuteLog, PostClassifier, host_of, run
from src.stream.sources.filters import load_live_config
from tests.test_live_source import SOURCE_CFG, FakeProducer

LIVE_SOURCES = load_live_config()
ENDPOINTS = LIVE_SOURCES["bluesky"]["endpoints"]
START = 1_800_000_000.0  # invented wall-clock start, in seconds
STEP = 0.5  # simulated seconds between two events


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class FakeTime:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def wall(self) -> float:
        return START + self.elapsed


class LaggingSocket:
    """Delivers one English post per STEP; its createdAt is `lag(elapsed)` seconds old."""

    def __init__(self, jetstream, lag):
        self.jetstream, self.lag = jetstream, lag

    async def recv(self):
        js = self.jetstream
        js.time.elapsed += STEP
        js.posts += 1
        if js.posts == js.signal_after:
            signal.raise_signal(signal.SIGTERM)
        now = js.time.wall()
        return json.dumps(
            {
                "did": "did:plc:testauthor0000000000000",
                "time_us": int(now * 1_000_000),
                "kind": "commit",
                "commit": {
                    "operation": "create",
                    "collection": "app.bsky.feed.post",
                    "rkey": f"3kfake{js.posts:06d}",
                    "record": {"text": "An invented post", "langs": ["en"], "createdAt": iso(now - self.lag(js.time.elapsed))},
                },
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class LaggingJetstream:
    """`lags[i](elapsed)` is the source lag of endpoint i."""

    def __init__(self, lags, store, signal_after=None):
        self.lags, self.store, self.signal_after = lags, store, signal_after
        self.time, self.posts, self.urls, self.cursor_at_connect = FakeTime(), 0, [], []

    def connect(self, url, **kwargs):
        self.urls.append(url)
        self.cursor_at_connect.append(self.store.load())
        index = next(i for i, e in enumerate(ENDPOINTS) if url.startswith(e))
        return LaggingSocket(self, self.lags.get(index, lambda elapsed: 1.0))


def run_lagging(tmp_path, lags, minutes, signal_after=None, stop_signals=()):
    store = CursorStore(str(tmp_path / "cursor.json"))
    jetstream = LaggingJetstream(lags, store, signal_after)
    tracker = DeliveryTracker()
    counts = asyncio.run(
        run(
            SOURCE_CFG,
            LIVE_SOURCES,
            PostClassifier(LIVE_SOURCES),
            KafkaSink(FakeProducer(), "posts_live", tracker),
            tracker,
            store,
            duration_seconds=minutes * 60,
            connect=jetstream.connect,
            clock=jetstream.time.wall,
            monotonic=jetstream.time.monotonic,
            stop_signals=stop_signals,
        )
    )
    return counts, jetstream, store


def cursor_in(url):
    found = re.search(r"&cursor=(\d+)$", url)
    return int(found.group(1)) if found else None


def test_switch_settings_are_in_the_config():
    bluesky = LIVE_SOURCES["bluesky"]
    assert bluesky["source_lag_switch_seconds"] == 30
    assert bluesky["source_lag_switch_after_minutes"] == 2
    assert bluesky["endpoint_switch_cooldown_minutes"] == 5


def test_minute_source_lag_is_received_minus_created_at():
    minute = MinuteLog()
    for lag in (10, 200, 30):
        received = START + 100
        minute.add({"topic": "general", "received_at_ms": int(received * 1000), "created_at": iso(received - lag)}, 0)
    minute.add({"topic": "general", "received_at_ms": 0, "created_at": None}, 0)  # no createdAt: left out
    assert minute.source_lag() == pytest.approx(30)
    assert MinuteLog().source_lag() is None


def test_a_slow_endpoint_is_left_for_the_next_one_with_the_cursor(tmp_path):
    counts, jetstream, _ = run_lagging(tmp_path, {0: lambda elapsed: 200.0}, minutes=8)
    assert counts["endpoint_switches"] == 1
    (switch,) = counts["switches"]
    assert (switch["from"], switch["to"]) == (host_of(ENDPOINTS[0]), host_of(ENDPOINTS[1]))
    assert switch["after_seconds"] == pytest.approx(120, abs=1)  # two slow minutes
    assert "200.0 s > 30 s for 2 minutes" in switch["reason"]
    assert jetstream.urls[1].startswith(ENDPOINTS[1])
    saved = jetstream.cursor_at_connect[1]
    assert saved is not None and cursor_in(jetstream.urls[1]) == saved - 5_000_000
    assert counts["dropped_connections"] == counts["failed_connection_attempts"] == 0
    assert len(jetstream.urls) == 2  # endpoint 2 is fast: no further switch


def test_one_slow_minute_does_not_switch(tmp_path):
    counts, jetstream, _ = run_lagging(tmp_path, {0: lambda elapsed: 200.0 if elapsed < 60 else 1.0}, minutes=6)
    assert counts["endpoint_switches"] == 0
    assert len(jetstream.urls) == 1


def test_lag_below_the_limit_does_not_switch(tmp_path):
    counts, _, _ = run_lagging(tmp_path, {0: lambda elapsed: 29.0}, minutes=6)
    assert counts["endpoint_switches"] == 0


def test_no_second_switch_within_the_cooldown(tmp_path):
    every_endpoint_slow = {i: (lambda elapsed: 200.0) for i in range(len(ENDPOINTS))}
    counts, jetstream, _ = run_lagging(tmp_path, every_endpoint_slow, minutes=12.5)
    times = [s["after_seconds"] for s in counts["switches"]]
    # slow for 2 minutes at 2:00, then the 5-minute cooldown: 7:00, 12:00
    assert times == pytest.approx([120, 420, 720], abs=1)
    assert all(later - earlier >= 300 for earlier, later in zip(times, times[1:]))
    assert [host_of(u) for u in jetstream.urls] == [host_of(e) for e in ENDPOINTS[:4]]


def test_sigterm_flushes_and_saves_the_cursor(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    counts, jetstream, store = run_lagging(
        tmp_path, {}, minutes=60, signal_after=50, stop_signals=(signal.SIGTERM, signal.SIGINT)
    )
    assert counts["stopped_by"] == "signal"
    assert 50 <= counts["sent"] < 60  # stopped right after the signal, not at the end of the hour
    assert counts["sent"] == counts["received"] and counts["failed"] == 0
    last_sent_us = int((START + counts["sent"] * STEP) * 1_000_000)
    assert store.load() == counts["cursor_on_stop"] == last_sent_us
    assert signal.getsignal(signal.SIGTERM) == before  # the handler is removed again
