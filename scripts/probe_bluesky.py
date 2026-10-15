"""Feasibility probe: how many English Bluesky posts are about airlines or flying, and how fresh are they?

Usage (from the repo root):
    python -m scripts.probe_bluesky --minutes 60 [--sample-csv PATH]

Listens to the public Bluesky Jetstream (no account needed) for the given time
and keeps new posts (`commit.operation == "create"`) whose `langs` include
English. Each post is checked at two levels from config/live_sources.yaml:

  airline_strict  names an airline (keywords, or a mention of its official account)
  air_travel      talks about flying ("flight delayed", "baggage claim", ...)

Each run is appended to reports/bluesky_probe.json. That file holds counts and
timings only: no post text, no handle, no DID. The first matches per airline are
printed to the screen, and --sample-csv writes a random sample of `air_travel`
matches (text only, other people's handles masked) to a file that should be kept
outside the repository, for a manual check of the filter.

This is a probe. The pipeline, the producer and the dashboard do not use it.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import socket
import struct
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
SAMPLE_SIZE = 40
SAMPLE_SEED = 42
UNSPECIFIED = "unspecified"
MENTION = "app.bsky.richtext.facet#mention"


def load_live_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def phrase_pattern(phrase: str) -> str:
    """A phrase as a whole-word regex: not preceded by a word character or "@", not followed by a word character."""
    return r"(?<![\w@])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"


class PhraseMatcher:
    """Case-insensitive whole-word matching of a list of phrases."""

    def __init__(self, phrases: list[str]):
        self.phrases = list(phrases)
        self.patterns = [re.compile(phrase_pattern(phrase), flags=re.IGNORECASE) for phrase in self.phrases]

    def first_match(self, text: str) -> str | None:
        return next((phrase for phrase, pattern in zip(self.phrases, self.patterns) if pattern.search(text)), None)


def official_accounts(airline_cfg: dict) -> list[dict]:
    account = airline_cfg.get("official_account")
    return [account] if isinstance(account, dict) else []


class AirlineMatcher:
    """Which airlines a post names: by keyword, by "@official.handle", or by a mention facet of the official DID."""

    def __init__(self, airlines_cfg: dict[str, dict]):
        self.keywords = {
            airline: PhraseMatcher(
                spec["keywords"] + ["@" + account["handle"] for account in official_accounts(spec)]
            )
            for airline, spec in airlines_cfg.items()
        }
        self.dids = {
            account["did"]: airline for airline, spec in airlines_cfg.items() for account in official_accounts(spec)
        }

    @property
    def airlines(self) -> list[str]:
        return list(self.keywords)

    def airlines_in(self, text: str, mentioned_dids=()) -> list[str]:
        by_did = {self.dids[did] for did in mentioned_dids if did in self.dids}
        return [
            airline
            for airline, matcher in self.keywords.items()
            if airline in by_did or matcher.first_match(text) is not None
        ]

    def via_did_only(self, text: str, mentioned_dids=()) -> list[str]:
        """Airlines found through a mention facet but not through any keyword."""
        return [
            self.dids[did]
            for did in mentioned_dids
            if did in self.dids and self.keywords[self.dids[did]].first_match(text) is None
        ]


def mentioned_dids(record: dict) -> list[str]:
    """DIDs of the accounts a post mentions, from its rich-text facets."""
    dids = []
    for facet in record.get("facets") or []:
        for feature in (facet or {}).get("features") or []:
            if isinstance(feature, dict) and feature.get("$type") == MENTION and isinstance(feature.get("did"), str):
                dids.append(feature["did"])
    return dids


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


def clock_offset(server: str, timeout: float = 5.0) -> float | None:
    """Seconds to add to this machine's clock to get the time server's (SNTP); None if unreachable."""
    ntp_to_unix = 2208988800
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.settimeout(timeout)
            sent = time.time()
            udp.sendto(b"\x1b" + 47 * b"\0", (server, 123))
            data, _ = udp.recvfrom(48)
            back = time.time()
    except OSError:
        return None

    def timestamp(offset: int) -> float:
        seconds, fraction = struct.unpack("!II", data[offset : offset + 8])
        return seconds - ntp_to_unix + fraction / 2**32

    received_by_server, sent_by_server = timestamp(32), timestamp(40)
    return round(((received_by_server - sent) + (sent_by_server - back)) / 2, 3)


def summary(values: list[float], shift: float = 0.0) -> dict:
    if not values:
        return {"n": 0, "median_seconds": None, "p95_seconds": None}
    median, p95 = np.percentile(np.asarray(values) + shift, [50, 95])
    return {"n": len(values), "median_seconds": round(float(median), 3), "p95_seconds": round(float(p95), 3)}


HANDLE = re.compile(r"(?<![\w@])@[\w.-]+")


def mask_handles(text: str, keep: set[str]) -> str:
    """Replace other people's @handles with "@user"; keep the airline handles listed in `keep`."""

    def replace(match: re.Match) -> str:
        handle = match.group(0)
        if any(handle.lower() == kept or handle.lower().startswith(kept + ".") for kept in keep):
            return handle
        return "@user"

    return HANDLE.sub(replace, text)


class ReservoirSample:
    """A uniform random sample of fixed size from a stream (seeded, so a run is repeatable)."""

    def __init__(self, size: int, seed: int):
        self.size, self.random, self.seen, self.items = size, random.Random(seed), 0, []

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            slot = self.random.randrange(self.seen)
            if slot < self.size:
                self.items[slot] = item


class ProbeStats:
    """Counts for one run. Holds no text and no identifiers."""

    def __init__(self, airlines: list[str], phrases: list[str], language: str, tolerance_seconds: float):
        self.language, self.tolerance = language, tolerance_seconds
        self.events = 0
        self.events_by_kind: dict[str, int] = {}
        self.creates = 0
        self.language_posts = 0
        self.strict_by_airline = dict.fromkeys(airlines, 0)
        self.strict_posts = 0
        self.strict_via_mention_facet = 0
        self.travel_posts = 0
        self.travel_by_airline = dict.fromkeys([*airlines, UNSPECIFIED], 0)
        self.travel_by_phrase = dict.fromkeys(phrases, 0)
        self.both_levels = 0
        self.lag_created_at: list[float] = []  # receive time - createdAt, all posts in the language
        self.lag_time_us: list[float] = []  # receive time - Jetstream time_us, same posts
        self.created_at_unreadable = 0
        self.created_at_off_by_more_than_tolerance = 0

    def add(self, event: dict, received: float, airlines: AirlineMatcher, travel: PhraseMatcher):
        """Count one Jetstream event. For a post in the language return (airlines, travel phrase), else None."""
        self.events += 1
        kind = str(event.get("kind"))
        self.events_by_kind[kind] = self.events_by_kind.get(kind, 0) + 1
        commit = event.get("commit") or {}
        if kind != "commit" or commit.get("operation") != "create":
            return None
        self.creates += 1
        record = commit.get("record") or {}
        if not is_language(record.get("langs"), self.language):
            return None
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

        text, dids = record.get("text") or "", mentioned_dids(record)
        named = airlines.airlines_in(text, dids)
        if named:
            self.strict_posts += 1
            for airline in named:
                self.strict_by_airline[airline] += 1
            if airlines.via_did_only(text, dids):
                self.strict_via_mention_facet += 1
        phrase = travel.first_match(text)
        if phrase is not None:
            self.travel_posts += 1
            self.travel_by_phrase[phrase] += 1
            for tag in named or [UNSPECIFIED]:
                self.travel_by_airline[tag] += 1
            if named:
                self.both_levels += 1
        return named, phrase

    def report(self, minutes: float, offset: float | None) -> dict:
        hours = minutes / 60
        shift = offset or 0.0

        def per_hour(counts: dict[str, int]) -> dict[str, float]:
            return {key: round(count / hours, 1) for key, count in counts.items()}

        return {
            "events": self.events,
            "events_by_kind": dict(sorted(self.events_by_kind.items())),
            "create_posts": self.creates,
            "english_posts": self.language_posts,
            "english_posts_per_hour": round(self.language_posts / hours, 1),
            "airline_strict": {
                "matched_posts": self.strict_posts,
                "matched_posts_per_hour": round(self.strict_posts / hours, 1),
                "matched_posts_by_airline": self.strict_by_airline,
                "matched_posts_per_hour_by_airline": per_hour(self.strict_by_airline),
                "matched_only_through_a_mention_facet": self.strict_via_mention_facet,
            },
            "air_travel": {
                "matched_posts": self.travel_posts,
                "matched_posts_per_hour": round(self.travel_posts / hours, 1),
                "matched_posts_by_airline_or_unspecified": self.travel_by_airline,
                "matched_posts_per_hour_by_airline_or_unspecified": per_hour(self.travel_by_airline),
                "matched_posts_by_first_phrase": self.travel_by_phrase,
                "also_airline_strict": self.both_levels,
            },
            "source_lag": {
                "clock_offset_applied_seconds": offset,
                "receive_minus_created_at": summary(self.lag_created_at),
                "receive_minus_jetstream_time_us": summary(self.lag_time_us),
                "receive_minus_created_at_corrected": summary(self.lag_created_at, shift),
                "receive_minus_jetstream_time_us_corrected": summary(self.lag_time_us, shift),
            },
            "created_at_unreadable": self.created_at_unreadable,
            "created_at_more_than_tolerance_from_time_us": self.created_at_off_by_more_than_tolerance,
        }


def show(airline: str, text: str, shown: dict[str, int]) -> None:
    """Print a matching post's text (never stored) for the first few matches of each airline."""
    if shown[airline] < SHOWN_PER_AIRLINE:
        shown[airline] += 1
        print(f"[{airline} {shown[airline]}/{SHOWN_PER_AIRLINE}] {' '.join(text.split())[:240]}", flush=True)


async def listen(cfg, minutes, stats, airlines, travel, sample, keep_handles) -> dict:
    """Read Jetstream until the time is up, reconnecting (with a cursor) when the connection drops."""
    endpoints = cfg["bluesky"]["endpoints"]
    deadline = time.monotonic() + minutes * 60
    shown = dict.fromkeys(airlines.airlines, 0)
    # dropped: an open connection that broke; failed attempts: connections that never opened
    connection = {"endpoints_used": [], "connections_dropped": 0, "reconnects": 0, "connection_attempts_failed": 0}
    cursor, attempt, connected_before = None, 0, False
    while time.monotonic() < deadline:
        endpoint = endpoints[attempt % len(endpoints)]
        url = endpoint + (f"&cursor={cursor}" if cursor else "")
        opened = False
        try:
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
                    matched = stats.add(event, received, airlines, travel)
                    if matched is None:
                        continue
                    named, phrase = matched
                    text = (event["commit"].get("record") or {}).get("text") or ""
                    for airline in named:
                        show(airline, text, shown)
                    if phrase is not None and sample is not None:
                        sample.offer({"text": mask_handles(" ".join(text.split()), keep_handles), "matched_phrase": phrase})
                return connection
        except (websockets.exceptions.WebSocketException, OSError, ConnectionError, asyncio.TimeoutError) as error:
            connection["connections_dropped" if opened else "connection_attempts_failed"] += 1
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
        "`python -m scripts.probe_bluesky`, one entry per run. The first run was written by an "
        "earlier version: one keyword level only, no clock offset, and `disconnects` counted every "
        "failure after the first connection, including reconnection attempts that never opened."
    )
    report["runs"].append(run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def write_sample(sample: ReservoirSample, path: str) -> None:
    """The review file: text, matched phrase and an empty `relevant` column. Keep it outside the repository."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel shows the text
        writer = csv.DictWriter(f, fieldnames=["text", "matched_phrase", "relevant"])
        writer.writeheader()
        for item in sample.items:
            writer.writerow({**item, "relevant": ""})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--output", default=REPORT_PATH)
    parser.add_argument("--sample-csv", default=None, help="where to write the review sample (outside the repo)")
    args = parser.parse_args()
    if args.sample_csv and os.path.abspath(args.sample_csv).startswith(os.path.abspath(config.BASE_DIR) + os.sep):
        parser.error("--sample-csv holds post text and must be outside the repository")

    cfg = load_live_config()
    airlines = AirlineMatcher(cfg["airline_strict"])
    travel = PhraseMatcher(cfg["air_travel"])
    stats = ProbeStats(airlines.airlines, travel.phrases, cfg["bluesky"]["language"],
                       cfg["bluesky"]["created_at_tolerance_minutes"] * 60)  # fmt: skip
    sample = ReservoirSample(SAMPLE_SIZE, SAMPLE_SEED) if args.sample_csv else None
    keep_handles = {
        phrase.lower() for matcher in airlines.keywords.values() for phrase in matcher.phrases if phrase.startswith("@")
    }

    server = cfg["bluesky"]["time_server"]
    offset_start = clock_offset(server)
    print(f"Clock offset against {server}: {offset_start} s", flush=True)
    started_wall, started = datetime.now(timezone.utc), time.monotonic()
    try:
        connection = asyncio.run(listen(cfg, args.minutes, stats, airlines, travel, sample, keep_handles))
    except KeyboardInterrupt:
        connection = {"interrupted": True}
    minutes = (time.monotonic() - started) / 60
    ended_wall = datetime.now(timezone.utc)
    offset_end = clock_offset(server)
    offsets = [o for o in (offset_start, offset_end) if o is not None]
    offset = round(sum(offsets) / len(offsets), 3) if offsets else None

    run = {
        "started_at_utc": started_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at_utc": ended_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "minutes_requested": args.minutes,
        "minutes_actual": round(minutes, 2),
        "clock": {
            "time_server": server,
            "offset_seconds_at_start": offset_start,
            "offset_seconds_at_end": offset_end,
            "meaning": "seconds to add to this machine's clock to get the time server's",
        },
        "connection": connection,
        "language": cfg["bluesky"]["language"],
        "keywords": {
            "airline_strict": {airline: matcher.phrases for airline, matcher in airlines.keywords.items()},
            "airline_strict_official_dids_used": len(airlines.dids),
            "air_travel": travel.phrases,
        },
        **stats.report(minutes, offset),
    }
    append_run(run, args.output)
    if sample is not None:
        write_sample(sample, args.sample_csv)
        print(f"Wrote {len(sample.items)} of {sample.seen} air_travel matches to {args.sample_csv}")
    print(json.dumps({"minutes_actual": run["minutes_actual"], "english_posts": run["english_posts"],
                      "airline_strict_per_hour": run["airline_strict"]["matched_posts_per_hour"],
                      "air_travel_per_hour": run["air_travel"]["matched_posts_per_hour"]}))  # fmt: skip
    print(f"Appended this run to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
