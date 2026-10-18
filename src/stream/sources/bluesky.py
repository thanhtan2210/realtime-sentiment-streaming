"""Live source: English posts from the public Bluesky Jetstream, sent to Kafka as they are written.

Run with `python -m src.stream.producer --source bluesky [--duration-minutes N]`.

Privacy: the author's DID and handle never leave this module. A post is keyed by
`post_id`, the sha256 of its AT URI (at://{did}/app.bsky.feed.post/{rkey}), and
other people's @handles in the text are replaced with "@user" (airline handles
are kept). The filters and the airline keywords come from config/live_sources.yaml.

Delivery: after every flush of the Kafka producer, the Jetstream `time_us` of the
last post that was acknowledged is saved as the cursor. On start-up, after a
dropped connection and after an endpoint switch, reading resumes from that
cursor minus a few seconds, so no post is skipped; the few posts read twice keep
their `post_id` and are written once by the processor.

Slow endpoints: every minute, the median source lag (received_at - the post's
createdAt) of that minute's posts is checked. When it is above
`source_lag_switch_seconds` for `source_lag_switch_after_minutes` minutes in a
row, the connection is closed and the next endpoint is used, at most once per
`endpoint_switch_cooldown_minutes` (config/live_sources.yaml).

Stopping: SIGTERM (`docker compose stop`) and SIGINT (Ctrl-C) end the run
cleanly: the producer is flushed, the cursor saved and a `LIVE_SUMMARY {json}`
line logged with aggregate counts only.
"""
import asyncio
import hashlib
import json
import os
import random
import signal
import statistics
import time
from datetime import datetime, timezone

import websockets

from src.common import config, logger as common_logger
from src.stream.sources.filters import (
    AirlineMatcher,
    PhraseMatcher,
    clock_offset,
    is_language,
    load_live_config,
    mask_handles,
    mentioned_dids,
    parse_created_at,
)

logger = common_logger.get_logger(__name__)

POST_COLLECTION = "app.bsky.feed.post"
AIR_TRAVEL, GENERAL = "air_travel", "general"
MESSAGE_FIELDS = ["post_id", "text", "topic", "event_time", "created_at", "received_at_ms", "produced_at_ms"]


def post_id(did: str, rkey: str) -> str:
    """sha256 of the post's AT URI: stable, but it does not reveal the author."""
    return hashlib.sha256(f"at://{did}/{POST_COLLECTION}/{rkey}".encode("utf-8")).hexdigest()


def iso_from_epoch(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


class PostClassifier:
    """`airline:<name>` when a post names an airline, `air_travel` when it is about flying, else `general`."""

    def __init__(self, live_sources: dict):
        self.language = live_sources["bluesky"]["language"]
        self.airlines = AirlineMatcher(live_sources["airline_strict"])
        self.travel = PhraseMatcher(live_sources["air_travel"])
        self.keep_handles = {
            phrase.lower()
            for matcher in self.airlines.keywords.values()
            for phrase in matcher.phrases
            if phrase.startswith("@")
        }

    def topic(self, text: str, dids=()) -> str:
        named = self.airlines.airlines_in(text, dids)
        if named:
            return f"airline:{named[0]}"
        return AIR_TRAVEL if self.travel.first_match(text) is not None else GENERAL


def to_message(event: dict, classifier: PostClassifier, received_at_ms: int):
    """A new English post as (message without produced_at_ms, its Jetstream time_us); None otherwise."""
    commit = event.get("commit") or {}
    if event.get("kind") != "commit" or commit.get("operation") != "create":
        return None
    if commit.get("collection", POST_COLLECTION) != POST_COLLECTION:
        return None
    record = commit.get("record") or {}
    did, rkey, time_us = event.get("did"), commit.get("rkey"), event.get("time_us")
    if not (isinstance(did, str) and isinstance(rkey, str) and isinstance(time_us, int)):
        return None
    if not is_language(record.get("langs"), classifier.language):
        return None
    text = " ".join(str(record.get("text") or "").split())
    if not text:
        return None
    created_at = parse_created_at(record.get("createdAt"))
    message = {
        "post_id": post_id(did, rkey),
        "text": mask_handles(text, classifier.keep_handles),
        "topic": classifier.topic(text, mentioned_dids(record)),
        "event_time": iso_from_epoch(time_us / 1e6),
        "created_at": iso_from_epoch(created_at) if created_at is not None else None,
        "received_at_ms": received_at_ms,
    }
    return message, time_us


class CursorStore:
    """The last acknowledged Jetstream time_us, in a small JSON file (a Docker volume in compose)."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> int | None:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                value = json.load(f)["time_us"]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"not a positive integer: {value!r}")
            return value
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning(f"Ignoring an unreadable cursor file {self.path}: {error!r}")
            return None

    def save(self, time_us: int) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump({"time_us": time_us, "saved_at": iso_from_epoch(time.time())}, f)
        os.replace(temporary, self.path)  # atomic: a crash leaves the old or the new file, never half of one


def backoff_delay(attempt: int, rng: random.Random, cap: float = 60.0, base: float = 1.0) -> float:
    """Exponential backoff with full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
    return rng.uniform(0, min(cap, base * 2 ** min(attempt, 30)))


class DeliveryTracker:
    """Which time_us is safe to save: every post up to it was acknowledged by Kafka."""

    def __init__(self):
        self.sent = 0
        self.failed = 0
        self.newest_sent = None  # largest time_us handed to Kafka since the last checkpoint
        self.oldest_failed = None  # smallest time_us that Kafka rejected since the last checkpoint

    def handed_over(self, time_us: int) -> None:
        self.newest_sent = time_us if self.newest_sent is None else max(self.newest_sent, time_us)

    def acknowledged(self) -> None:
        self.sent += 1

    def rejected(self, time_us: int) -> None:
        self.failed += 1
        self.oldest_failed = time_us if self.oldest_failed is None else min(self.oldest_failed, time_us)

    def checkpoint(self) -> int | None:
        """After a flush: the cursor to save (None if nothing new), and a clean slate."""
        if self.oldest_failed is not None:
            cursor = self.oldest_failed - 1
        else:
            cursor = self.newest_sent
        self.newest_sent = self.oldest_failed = None
        return cursor


class KafkaSink:
    """Sends messages to Kafka, keyed by post_id, and reports deliveries to a DeliveryTracker."""

    def __init__(self, producer, topic: str, tracker: DeliveryTracker):
        self.producer, self.topic, self.tracker = producer, topic, tracker

    def send(self, message: dict, time_us: int) -> None:
        self.tracker.handed_over(time_us)
        future = self.producer.send(
            self.topic, key=message["post_id"].encode("utf-8"), value=json.dumps(message).encode("utf-8")
        )
        future.add_callback(lambda _metadata: self.tracker.acknowledged())
        future.add_errback(lambda error: self.tracker.rejected(time_us))

    def flush(self) -> None:
        self.producer.flush()


class MinuteLog:
    """Counts for the log line and the source-lag check, one minute at a time."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.received = 0
        self.sent_at_start = 0
        self.topics: dict[str, int] = {}
        self.created_lags: list[float] = []  # received_at - createdAt
        self.jetstream_lags: list[float] = []  # received_at - Jetstream time_us

    def add(self, message: dict, time_us: int) -> None:
        self.received += 1
        self.topics[message["topic"]] = self.topics.get(message["topic"], 0) + 1
        received = message["received_at_ms"] / 1000
        self.jetstream_lags.append(received - time_us / 1e6)
        if message["created_at"] is not None:
            self.created_lags.append(received - datetime.fromisoformat(message["created_at"]).timestamp())

    def source_lag(self) -> float | None:
        """Median received_at - createdAt of this minute's posts (None without posts)."""
        return statistics.median(self.created_lags) if self.created_lags else None

    def line(self, sent_total: int, counts: dict) -> str:
        lag = self.source_lag()
        jetstream = statistics.median(self.jetstream_lags) if self.jetstream_lags else None
        topics = dict(sorted(self.topics.items(), key=lambda item: -item[1]))
        return (
            f"LIVE received={self.received} sent={sent_total - self.sent_at_start} "
            f"reconnects={counts['reconnects']} endpoint_switches={counts['endpoint_switches']} "
            f"median_source_lag_s={None if lag is None else round(lag, 3)} "
            f"median_jetstream_lag_s={None if jetstream is None else round(jetstream, 3)} "
            f"topics={json.dumps(topics)}"
        )


class EndpointSwitch(Exception):
    """Raised inside the read loop to leave a slow endpoint."""


def host_of(endpoint: str) -> str:
    return endpoint.split("//", 1)[-1].split("/", 1)[0]


def install_stop_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event, signals) -> list:
    """Make each signal set `stop`; returns what is needed to undo it."""
    undo = []
    for sig in signals:
        try:
            loop.add_signal_handler(sig, stop.set)
            undo.append((sig, None))
        except (NotImplementedError, RuntimeError):  # Windows event loops
            previous = signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
            undo.append((sig, previous))
    return undo


def remove_stop_handlers(loop: asyncio.AbstractEventLoop, undo: list) -> None:
    for sig, previous in undo:
        if previous is None:
            loop.remove_signal_handler(sig)
        else:
            signal.signal(sig, previous)


async def run(
    source_cfg: dict,
    live_sources: dict,
    classifier: PostClassifier,
    sink,
    tracker: DeliveryTracker,
    store: CursorStore,
    duration_seconds: float | None = None,
    connect=websockets.connect,
    clock=time.time,
    monotonic=time.monotonic,
    rng: random.Random | None = None,
    stop_signals=(),
    minute_seconds: float = 60.0,
) -> dict:
    """Read Jetstream and send posts until the duration is up or a stop signal; returns the counts."""
    rng = rng or random.Random()
    bluesky = live_sources["bluesky"]
    endpoints = bluesky["endpoints"]
    lag_limit = bluesky["source_lag_switch_seconds"]
    slow_minutes_to_switch = bluesky["source_lag_switch_after_minutes"]
    cooldown = bluesky["endpoint_switch_cooldown_minutes"] * minute_seconds
    rewind_us = int(source_cfg["cursor_rewind_seconds"] * 1_000_000)
    started = monotonic()
    deadline = None if duration_seconds is None else started + duration_seconds
    minute = MinuteLog()
    counts = {
        "received": 0,
        "connections": 0,
        "reconnects": 0,
        "dropped_connections": 0,
        "failed_connection_attempts": 0,
        "endpoint_switches": 0,
        "switches": [],
        "stopped_by": "duration",
    }

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    undo = install_stop_handlers(loop, stop, stop_signals)
    stop_wait = asyncio.ensure_future(stop.wait())

    saved = store.load()
    counts["cursor_on_start"] = saved
    resume_from = saved - rewind_us if saved else None
    if saved:
        logger.info(f"Resuming from the saved cursor minus {source_cfg['cursor_rewind_seconds']} s")
    else:
        logger.info("No saved cursor: starting from now")

    def checkpoint() -> None:
        nonlocal saved
        sink.flush()
        cursor = tracker.checkpoint()
        if cursor is not None and (saved is None or cursor > saved):
            store.save(cursor)
            saved = cursor

    def remaining() -> float | None:
        return None if deadline is None else deadline - monotonic()

    def running() -> bool:
        return not stop.is_set() and (deadline is None or remaining() > 0)

    attempt, endpoint_index = 0, 0
    last_switch = None
    try:
        while running():
            endpoint = endpoints[endpoint_index % len(endpoints)]
            url = endpoint + (f"&cursor={resume_from}" if resume_from else "")
            opened = False
            try:
                async with connect(url, open_timeout=15, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                    opened = True
                    counts["connections"] += 1
                    counts["reconnects"] = counts["connections"] - 1
                    attempt = 0
                    slow_minutes = 0
                    minute.reset()
                    minute.sent_at_start = tracker.sent
                    last_checkpoint = minute_start = monotonic()
                    logger.info(f"Connected to {endpoint}" + (" (with a cursor)" if resume_from else ""))
                    while running():
                        timeout = 30 if remaining() is None else min(30, max(remaining(), 0.001))
                        receive = asyncio.ensure_future(ws.recv())
                        done, _ = await asyncio.wait(
                            {receive, stop_wait}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                        )
                        if receive not in done:
                            receive.cancel()
                            if not running():
                                break
                            raise ConnectionError("no event for 30 seconds")
                        raw = receive.result()
                        received_at_ms = int(clock() * 1000)
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            event = None  # not JSON: skip it
                        converted = to_message(event, classifier, received_at_ms) if isinstance(event, dict) else None
                        if converted is not None:
                            message, time_us = converted
                            message["produced_at_ms"] = int(clock() * 1000)
                            sink.send(message, time_us)
                            minute.add(message, time_us)
                            counts["received"] += 1
                        now = monotonic()
                        if now - last_checkpoint >= source_cfg["cursor_save_seconds"]:
                            checkpoint()
                            last_checkpoint = now
                        if now - minute_start >= minute_seconds:
                            logger.info(minute.line(tracker.sent, counts))
                            lag = minute.source_lag()
                            slow_minutes = slow_minutes + 1 if lag is not None and lag > lag_limit else 0
                            minute.reset()
                            minute.sent_at_start = tracker.sent
                            minute_start = now
                            cooled_down = last_switch is None or now - last_switch >= cooldown
                            if slow_minutes >= slow_minutes_to_switch and cooled_down:
                                raise EndpointSwitch(
                                    f"median source lag {lag:.1f} s > {lag_limit} s for {slow_minutes} minutes"
                                )
            except EndpointSwitch as switch:
                checkpoint()
                resume_from = saved - rewind_us if saved else resume_from
                endpoint_index += 1
                last_switch = monotonic()
                target = endpoints[endpoint_index % len(endpoints)]
                counts["endpoint_switches"] += 1
                counts["switches"].append(
                    {
                        "from": host_of(endpoint),
                        "to": host_of(target),
                        "reason": str(switch),
                        "after_seconds": round(last_switch - started, 1),
                    }
                )
                logger.warning(f"Switching from {host_of(endpoint)} to {host_of(target)}: {switch}")
            except (websockets.exceptions.WebSocketException, OSError, ConnectionError, asyncio.TimeoutError) as error:
                if opened:
                    counts["dropped_connections"] += 1
                else:
                    counts["failed_connection_attempts"] += 1
                logger.warning(f"Connection to {endpoint} lost or refused: {error!r}")
                checkpoint()
                resume_from = saved - rewind_us if saved else resume_from
                delay = backoff_delay(attempt, rng, cap=source_cfg["backoff_cap_seconds"])
                attempt += 1
                endpoint_index += 1
                if running():
                    logger.info(f"Reconnecting in {delay:.1f} s")
                    await asyncio.wait({stop_wait}, timeout=delay)
    finally:
        if stop.is_set():
            counts["stopped_by"] = "signal"
        stop_wait.cancel()
        remove_stop_handlers(loop, undo)
        checkpoint()
    counts.update({"sent": tracker.sent, "failed": tracker.failed, "cursor_on_stop": saved})
    return counts


def main(duration_minutes: float | None, live_cfg: dict) -> int:
    """Entry point for `producer --source bluesky`."""
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    live_sources = load_live_config()
    source_cfg = dict(live_cfg["source"])
    cursor_path = source_cfg["cursor_path"]
    if not os.path.isabs(cursor_path):
        cursor_path = os.path.join(config.BASE_DIR, cursor_path)
    kafka = live_cfg["kafka"]
    try:
        producer = KafkaProducer(bootstrap_servers=kafka["bootstrap_servers"], acks="all", linger_ms=20)
    except KafkaError as error:
        logger.error(f"Cannot connect to Kafka at {kafka['bootstrap_servers']}: {error!r}")
        return 1

    time_server = live_sources["bluesky"]["time_server"]
    summary = {"started_at": iso_from_epoch(time.time()), "clock_offset_start_s": clock_offset(time_server)}
    tracker = DeliveryTracker()
    sink = KafkaSink(producer, kafka["topic"], tracker)
    duration = None if duration_minutes is None else duration_minutes * 60
    logger.info(f"Sending English Bluesky posts to '{kafka['topic']}'" + (f" for {duration_minutes} min" if duration else ""))
    try:
        counts = asyncio.run(
            run(
                source_cfg,
                live_sources,
                PostClassifier(live_sources),
                sink,
                tracker,
                CursorStore(cursor_path),
                duration,
                stop_signals=(signal.SIGTERM, signal.SIGINT),
            )
        )
    finally:
        producer.close()
    summary.update(counts)
    summary.update({"stopped_at": iso_from_epoch(time.time()), "clock_offset_end_s": clock_offset(time_server)})
    logger.info("LIVE_SUMMARY " + json.dumps(summary))
    return 0 if counts["failed"] == 0 else 1
