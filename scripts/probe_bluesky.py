"""Feasibility probe: how many English Bluesky posts mention an airline, and how fresh are they?

Usage (from the repo root):
    python -m scripts.probe_bluesky --minutes 60

Listens to the public Bluesky Jetstream (no account needed) for the given time,
keeps new posts (`commit.operation == "create"`) whose `langs` include English,
and matches them against the airline keywords in config/live_sources.yaml.

Each run is appended to reports/bluesky_probe.json. The file holds counts and
timings only: no post text, no handle, no DID. The first matches per airline
are printed to the screen (text only) so the keywords can be checked by eye.

This is a probe. The pipeline, the producer and the dashboard do not use it.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import websockets
import yaml

from src.common import config

CONFIG_PATH = os.path.join(config.BASE_DIR, "config", "live_sources.yaml")
REPORT_PATH = os.path.join(config.BASE_DIR, "reports", "bluesky_probe.json")
SHOWN_PER_AIRLINE = 5


def load_live_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class AirlineMatcher:
    """Case-insensitive keyword matching on whole words; "@handle" keywords also match "@handle.domain"."""

    def __init__(self, keywords: dict[str, list[str]]):
        self.patterns = {
            airline: re.compile(
                "|".join(
                    # not preceded by a word character or "@", not followed by a word character
                    r"(?<![\w@])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"
                    for phrase in phrases
                ),
                flags=re.IGNORECASE,
            )
            for airline, phrases in keywords.items()
        }

    def airlines_in(self, text: str) -> list[str]:
        return [airline for airline, pattern in self.patterns.items() if pattern.search(text)]


def is_language(langs, code: str) -> bool:
    return any(lang == code or lang.startswith(code + "-") for lang in langs or [] if isinstance(lang, str))


def parse_created_at(value) -> float | None:
    """createdAt in seconds since the epoch; None if missing or unreadable (it is written by clients)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median_seconds": None, "p95_seconds": None}
    median, p95 = np.percentile(values, [50, 95])
    return {"n": len(values), "median_seconds": round(float(median), 3), "p95_seconds": round(float(p95), 3)}


class ProbeStats:
    """Counts for one run. Holds no text and no identifiers."""

    def __init__(self, airlines: list[str], language: str, tolerance_seconds: float):
        self.language, self.tolerance = language, tolerance_seconds
        self.events = 0
        self.events_by_kind: dict[str, int] = {}
        self.creates = 0
        self.language_posts = 0
        self.matched_by_airline = dict.fromkeys(airlines, 0)
        self.matched_posts = 0
        self.lag_created_at: list[float] = []  # receive time - createdAt, all posts in the language
        self.lag_time_us: list[float] = []  # receive time - Jetstream time_us, same posts
        self.matched_lag_created_at: list[float] = []
        self.matched_lag_time_us: list[float] = []
        self.created_at_unreadable = 0
        self.created_at_off_by_more_than_tolerance = 0

    def add(self, event: dict, received: float, matcher: AirlineMatcher) -> list[str]:
        """Count one Jetstream event; return the airlines a matching post mentions."""
        self.events += 1
        kind = str(event.get("kind"))
        self.events_by_kind[kind] = self.events_by_kind.get(kind, 0) + 1
        commit = event.get("commit") or {}
        if kind != "commit" or commit.get("operation") != "create":
            return []
        self.creates += 1
        record = commit.get("record") or {}
        if not is_language(record.get("langs"), self.language):
            return []
        self.language_posts += 1

        server_time = event["time_us"] / 1e6 if isinstance(event.get("time_us"), (int, float)) else None
        created_at = parse_created_at(record.get("createdAt"))
        if server_time is not None:
            self.lag_time_us.append(received - server_time)
        if created_at is None:
            self.created_at_unreadable += 1
        else:
            self.lag_created_at.append(received - created_at)
            if server_time is not None and abs(created_at - server_time) > self.tolerance:
                self.created_at_off_by_more_than_tolerance += 1

        airlines = matcher.airlines_in(record.get("text") or "")
        if airlines:
            self.matched_posts += 1
            for airline in airlines:
                self.matched_by_airline[airline] += 1
            if server_time is not None:
                self.matched_lag_time_us.append(received - server_time)
            if created_at is not None:
                self.matched_lag_created_at.append(received - created_at)
        return airlines

    def report(self, minutes: float) -> dict:
        hours = minutes / 60
        return {
            "events": self.events,
            "events_by_kind": dict(sorted(self.events_by_kind.items())),
            "create_posts": self.creates,
            "english_posts": self.language_posts,
            "matched_posts_by_airline": self.matched_by_airline,
            "matched_posts": self.matched_posts,
            "matched_posts_per_hour_by_airline": {
                airline: round(count / hours, 1) for airline, count in self.matched_by_airline.items()
            },
            "matched_posts_per_hour": round(self.matched_posts / hours, 1),
            "source_lag": {
                "receive_minus_created_at": summary(self.lag_created_at),
                "receive_minus_jetstream_time_us": summary(self.lag_time_us),
                "matched_receive_minus_created_at": summary(self.matched_lag_created_at),
                "matched_receive_minus_jetstream_time_us": summary(self.matched_lag_time_us),
            },
            "created_at_unreadable": self.created_at_unreadable,
            "created_at_more_than_tolerance_from_time_us": self.created_at_off_by_more_than_tolerance,
        }


def show(airline: str, text: str, shown: dict[str, int]) -> None:
    """Print a matching post's text (never stored) for the first few matches of each airline."""
    if shown[airline] < SHOWN_PER_AIRLINE:
        shown[airline] += 1
        print(f"[{airline} {shown[airline]}/{SHOWN_PER_AIRLINE}] {' '.join(text.split())[:240]}", flush=True)


async def listen(cfg: dict, minutes: float, stats: ProbeStats, matcher: AirlineMatcher) -> dict:
    """Read Jetstream until the time is up, reconnecting (with a cursor) when the connection drops."""
    endpoints = cfg["bluesky"]["endpoints"]
    deadline = time.monotonic() + minutes * 60
    shown = dict.fromkeys(matcher.patterns, 0)
    # dropped: an open connection that broke; failed attempts: connections that never opened
    connection = {"endpoints_used": [], "connections_dropped": 0, "reconnects": 0, "connection_attempts_failed": 0}
    cursor, attempt, connected_before, opened = None, 0, False, False
    while time.monotonic() < deadline:
        endpoint = endpoints[attempt % len(endpoints)]
        url = endpoint + (f"&cursor={cursor}" if cursor else "")
        try:
            opened = False
            async with websockets.connect(url, open_timeout=15, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                opened = True
                if connected_before:
                    connection["reconnects"] += 1
                connected_before = True
                if endpoint not in connection["endpoints_used"]:
                    connection["endpoints_used"].append(endpoint)
                print(f"Connected to {endpoint}", flush=True)
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
                    except asyncio.TimeoutError:
                        if deadline - time.monotonic() <= 0:
                            break
                        raise ConnectionError("no event for 30 seconds")
                    received = time.time()
                    event = json.loads(raw)
                    if isinstance(event.get("time_us"), int):
                        cursor = event["time_us"]
                    for airline in stats.add(event, received, matcher):
                        show(airline, (event["commit"].get("record") or {}).get("text") or "", shown)
                return connection
        except (websockets.exceptions.WebSocketException, OSError, ConnectionError, asyncio.TimeoutError) as error:
            if opened:
                connection["connections_dropped"] += 1
            else:
                connection["connection_attempts_failed"] += 1
            print(f"Connection to {endpoint} lost or refused: {error!r}", flush=True)
            attempt += 1
            if not connected_before and attempt >= len(endpoints):
                raise SystemExit("Could not connect to any Jetstream endpoint in config/live_sources.yaml")
            await asyncio.sleep(min(30, 2 ** min(attempt, 5)))
    return connection


def append_run(run: dict, path: str = REPORT_PATH) -> None:
    report = {"runs": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    report["note"] = (
        "Counts and timings only; no post text, handle or DID is stored. Written by "
        "`python -m scripts.probe_bluesky`, one entry per run. The first run was written by a "
        "version that counted every failure after the first connection as `disconnects`, "
        "including reconnection attempts that never opened."
    )
    report["runs"].append(run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--output", default=REPORT_PATH)
    args = parser.parse_args()

    cfg = load_live_config()
    matcher = AirlineMatcher(cfg["airlines"])
    stats = ProbeStats(list(cfg["airlines"]), cfg["bluesky"]["language"],
                       cfg["bluesky"]["created_at_tolerance_minutes"] * 60)  # fmt: skip
    started_wall, started = datetime.now(timezone.utc), time.monotonic()
    try:
        connection = asyncio.run(listen(cfg, args.minutes, stats, matcher))
    except KeyboardInterrupt:
        connection = {"interrupted": True}
    minutes = (time.monotonic() - started) / 60
    run = {
        "started_at_utc": started_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "minutes_requested": args.minutes,
        "minutes_actual": round(minutes, 2),
        "connection": connection,
        "language": cfg["bluesky"]["language"],
        "keywords": cfg["airlines"],
        **stats.report(minutes),
    }
    append_run(run, args.output)
    print(json.dumps({k: run[k] for k in ("minutes_actual", "english_posts", "matched_posts_by_airline",
                                          "matched_posts_per_hour")}))  # fmt: skip
    print(f"Appended this run to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
