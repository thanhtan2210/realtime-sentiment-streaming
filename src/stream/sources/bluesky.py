"""Live source: English posts from the public Bluesky Jetstream, sent to Kafka as they are written.

Run with `python -m src.stream.producer --source bluesky [--duration-minutes N]`.

Privacy: the author's DID and handle never leave this module. A post is keyed by
`post_id`, the sha256 of its AT URI (at://{did}/app.bsky.feed.post/{rkey}), and
other people's @handles in the text are replaced with "@user" (airline handles
are kept). The filters and the airline keywords come from config/live_sources.yaml.

Delivery: after every flush of the Kafka producer, the Jetstream `time_us` of the
last post that was acknowledged is saved as the cursor. On start-up, and after a
dropped connection, reading resumes from that cursor minus a few seconds, so no
post is skipped; the few posts read twice keep their `post_id` and are written
once by the processor.
"""
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone

import websockets

from src.common import config, logger as common_logger
from src.stream.sources.filters import (
    AirlineMatcher,
    PhraseMatcher,
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
    """Counts for the periodic log line."""

    def __init__(self):
        self.reset()
        self.reconnects = 0

    def reset(self) -> None:
        self.received = 0
        self.topics: dict[str, int] = {}
        self.lags: list[float] = []
        self.sent_at_start = None

    def add(self, message: dict, time_us: int) -> None:
        self.received += 1
        self.topics[message["topic"]] = self.topics.get(message["topic"], 0) + 1
        self.lags.append(message["received_at_ms"] / 1000 - time_us / 1e6)

    def line(self, sent_total: int) -> str:
        sent = sent_total - (self.sent_at_start or 0)
        lag = round(statistics.median(self.lags), 3) if self.lags else None
        topics = dict(sorted(self.topics.items(), key=lambda item: -item[1]))
        return (
            f"LIVE received={self.received} sent={sent} reconnects={self.reconnects} "
            f"median_source_lag_s={lag} topics={json.dumps(topics)}"
        )


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
    rng: random.Random | None = None,
) -> dict:
    """Read Jetstream and send posts until the duration is up; returns the final counts."""
    rng = rng or random.Random()
    endpoints = live_sources["bluesky"]["endpoints"]
    rewind_us = int(source_cfg["cursor_rewind_seconds"] * 1_000_000)
    deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
    minute = MinuteLog()
    minute.sent_at_start = tracker.sent
    counts = {"connections": 0, "reconnects": 0, "failed_connection_attempts": 0, "cursor_on_start": None}

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
        return None if deadline is None else deadline - time.monotonic()

    attempt, endpoint_index = 0, 0
    last_checkpoint = last_log = time.monotonic()
    try:
        while remaining() is None or remaining() > 0:
            endpoint = endpoints[endpoint_index % len(endpoints)]
            url = endpoint + (f"&cursor={resume_from}" if resume_from else "")
            opened = False
            try:
                async with connect(url, open_timeout=15, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                    opened = True
                    counts["connections"] += 1
                    if counts["connections"] > 1:
                        counts["reconnects"] += 1
                        minute.reconnects += 1
                    attempt = 0
                    logger.info(f"Connected to {endpoint}" + (" (with a cursor)" if resume_from else ""))
                    while remaining() is None or remaining() > 0:
                        timeout = 30 if remaining() is None else min(30, remaining())
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            if remaining() is not None and remaining() <= 0:
                                break
                            raise ConnectionError("no event for 30 seconds")
                        received_at_ms = int(clock() * 1000)
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            continue  # not JSON: skip it
                        converted = to_message(event, classifier, received_at_ms) if isinstance(event, dict) else None
                        if converted is not None:
                            message, time_us = converted
                            message["produced_at_ms"] = int(clock() * 1000)
                            sink.send(message, time_us)
                            minute.add(message, time_us)
                        now = time.monotonic()
                        if now - last_checkpoint >= source_cfg["cursor_save_seconds"]:
                            checkpoint()
                            last_checkpoint = now
                        if now - last_log >= source_cfg["log_every_seconds"]:
                            logger.info(minute.line(tracker.sent))
                            minute.reset()
                            minute.sent_at_start = tracker.sent
                            last_log = now
                    break  # the duration is up
            except (websockets.exceptions.WebSocketException, OSError, ConnectionError, asyncio.TimeoutError) as error:
                if not opened:
                    counts["failed_connection_attempts"] += 1
                logger.warning(f"Connection to {endpoint} lost or refused: {error!r}")
                checkpoint()
                resume_from = saved - rewind_us if saved else resume_from
                delay = backoff_delay(attempt, rng, cap=source_cfg["backoff_cap_seconds"])
                attempt += 1
                endpoint_index += 1
                logger.info(f"Reconnecting in {delay:.1f} s")
                await asyncio.sleep(delay)
    finally:
        checkpoint()
        logger.info(minute.line(tracker.sent))
    counts.update({"sent": tracker.sent, "failed": tracker.failed, "cursor_on_stop": saved})
    return counts


def main(duration_minutes: float | None, live_cfg: dict) -> int:
    """Entry point for `producer --source bluesky`."""
    import signal

    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    # `docker compose stop` sends SIGTERM: treat it like Ctrl-C, so the last
    # posts are flushed and the cursor is saved before the process exits.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

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

    tracker = DeliveryTracker()
    sink = KafkaSink(producer, kafka["topic"], tracker)
    duration = None if duration_minutes is None else duration_minutes * 60
    logger.info(f"Sending English Bluesky posts to '{kafka['topic']}'" + (f" for {duration_minutes} min" if duration else ""))
    try:
        counts = asyncio.run(
            run(source_cfg, live_sources, PostClassifier(live_sources), sink, tracker, CursorStore(cursor_path), duration)
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 0
    finally:
        producer.close()
    logger.info(f"Stopped: {json.dumps(counts)}")
    return 0 if counts["failed"] == 0 else 1
