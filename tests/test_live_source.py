"""The live Bluesky source, without any network: a fake Jetstream and invented posts.

Every DID, rkey and text below is made up for these tests.
"""
import asyncio
import hashlib
import json
import random
import re

import pytest
import websockets

from src.stream.producer import parse_args
from src.stream.sources.bluesky import (
    MESSAGE_FIELDS,
    CursorStore,
    DeliveryTracker,
    KafkaSink,
    PostClassifier,
    backoff_delay,
    post_id,
    run,
    to_message,
)
from src.stream.sources.filters import load_live_config

LIVE_SOURCES = load_live_config()
FAKE_DID = "did:plc:testauthor0000000000000"
FAKE_HANDLE = "someone.example.test"
UNITED_DID = LIVE_SOURCES["airline_strict"]["United"]["official_account"]["did"]
T0 = 1_800_000_000_000_000  # an invented Jetstream time_us
SOURCE_CFG = {
    "cursor_path": "unused",
    "cursor_rewind_seconds": 5,
    "cursor_save_seconds": 0,  # flush and save after every event
    "backoff_cap_seconds": 0.01,
}


@pytest.fixture(scope="module")
def classifier():
    return PostClassifier(LIVE_SOURCES)


def event(rkey="3kfakepost0001", text="Lovely morning for a walk", langs=("en",), time_us=T0, operation="create",
          collection="app.bsky.feed.post", kind="commit", did=FAKE_DID, facets=None):
    record = {"$type": collection, "text": text, "langs": list(langs), "createdAt": "2027-01-15T08:00:00.000Z"}
    if facets:
        record["facets"] = facets
    return {
        "did": did,
        "time_us": time_us,
        "kind": kind,
        "commit": {"operation": operation, "collection": collection, "rkey": rkey, "record": record},
    }


def mention(did):
    return [{"index": {"byteStart": 0, "byteEnd": 4}, "features": [{"$type": "app.bsky.richtext.facet#mention", "did": did}]}]


# --- filtering and the message ----------------------------------------------------------------


@pytest.mark.parametrize("langs", [("en",), ("en-US",), ("en-GB", "fr"), ("de", "en")])
def test_english_posts_are_kept(classifier, langs):
    assert to_message(event(langs=langs), classifier, 0) is not None


@pytest.mark.parametrize("langs", [(), ("fr",), ("eng",), ("ja", "pt-BR")])
def test_other_languages_are_dropped(classifier, langs):
    assert to_message(event(langs=langs), classifier, 0) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "delete"},
        {"operation": "update"},
        {"kind": "identity"},
        {"kind": "account"},
        {"collection": "app.bsky.feed.like"},
        {"text": "   "},
    ],
)
def test_only_new_posts_with_text_are_kept(classifier, changes):
    assert to_message(event(**changes), classifier, 0) is None


def test_post_id_is_the_sha256_of_the_at_uri():
    expected = hashlib.sha256(f"at://{FAKE_DID}/app.bsky.feed.post/3kfakepost0001".encode("utf-8")).hexdigest()
    assert post_id(FAKE_DID, "3kfakepost0001") == expected
    assert post_id(FAKE_DID, "3kfakepost0002") != expected


def test_message_fields_and_times(classifier):
    message, time_us = to_message(event(time_us=T0), classifier, received_at_ms=1_800_000_000_250)
    assert time_us == T0
    assert set(message) == set(MESSAGE_FIELDS) - {"produced_at_ms"}
    assert message["post_id"] == post_id(FAKE_DID, "3kfakepost0001")
    assert message["event_time"] == "2027-01-15T08:00:00.000+00:00"
    assert message["created_at"] == "2027-01-15T08:00:00.000+00:00"
    assert message["received_at_ms"] == 1_800_000_000_250


@pytest.mark.parametrize(
    "text, facets, topic",
    [
        ("Two hours on hold with United Airlines again", None, "airline:United"),
        ("thanks @southwestair, smooth trip", None, "airline:Southwest"),
        ("Booked the trip, fingers crossed", mention(UNITED_DID), "airline:United"),
        ("my flight was delayed by three hours", None, "air_travel"),
        ("Sunset over the river delta tonight", None, "general"),
    ],
)
def test_topic(classifier, text, facets, topic):
    message, _ = to_message(event(text=text, facets=facets), classifier, 0)
    assert message["topic"] == topic


def test_message_has_no_did_or_handle(classifier):
    text = f"@{FAKE_HANDLE} lost my luggage again, thanks @united"
    message, _ = to_message(event(text=text, facets=mention(FAKE_DID)), classifier, 0)
    serialized = json.dumps(message)
    assert FAKE_DID not in serialized and "did:" not in serialized
    assert FAKE_HANDLE not in serialized
    assert message["text"] == "@user lost my luggage again, thanks @united"  # airline handles are kept
    assert message["topic"] == "airline:United"


# --- cursor, backoff, delivery ----------------------------------------------------------------


def test_cursor_round_trip(tmp_path):
    store = CursorStore(str(tmp_path / "state" / "cursor.json"))
    assert store.load() is None
    store.save(T0)
    assert store.load() == T0
    store.save(T0 + 1)
    assert store.load() == T0 + 1
    assert not (tmp_path / "state" / "cursor.json.tmp").exists()


@pytest.mark.parametrize("content", ["", "{not json", '{"time_us": "abc"}', '{"time_us": -5}', '{"other": 1}', "[1, 2]"])
def test_unreadable_cursor_means_start_from_now(tmp_path, content):
    path = tmp_path / "cursor.json"
    path.write_text(content, encoding="utf-8")
    assert CursorStore(str(path)).load() is None


def test_backoff_is_exponential_jittered_and_capped():
    rng = random.Random(0)
    for attempt in range(12):
        bound = min(60.0, 2.0**attempt)
        delays = [backoff_delay(attempt, rng) for _ in range(200)]
        assert all(0 <= d <= bound for d in delays)
        assert max(delays) > bound / 2  # jitter spreads over the whole range
    assert backoff_delay(10_000, rng) <= 60.0


def test_cursor_stops_before_a_rejected_post():
    tracker = DeliveryTracker()
    for time_us in (T0, T0 + 10, T0 + 20):
        tracker.handed_over(time_us)
    tracker.rejected(T0 + 10)
    assert tracker.checkpoint() == T0 + 9
    assert tracker.checkpoint() is None  # nothing new since


class FakeFuture:
    def __init__(self, ok):
        self.ok, self.callbacks, self.errbacks = ok, [], []

    def add_callback(self, f):
        self.callbacks.append(f)

    def add_errback(self, f):
        self.errbacks.append(f)

    def resolve(self):
        for f in self.callbacks if self.ok else self.errbacks:
            f(None if self.ok else RuntimeError("rejected"))


class FakeProducer:
    """Acknowledges (or rejects) every message on flush, like a Kafka producer would."""

    def __init__(self, reject_post_ids=()):
        self.reject, self.records, self.pending = set(reject_post_ids), [], []

    def send(self, topic, key, value):
        message = json.loads(value)
        future = FakeFuture(message["post_id"] not in self.reject)
        self.records.append((topic, key, message))
        self.pending.append(future)
        return future

    def flush(self):
        for future in self.pending:
            future.resolve()
        self.pending = []


def test_kafka_sink_keys_by_post_id():
    producer, tracker = FakeProducer(), DeliveryTracker()
    sink = KafkaSink(producer, "posts_live", tracker)
    sink.send({"post_id": "abc", "text": "hi"}, T0)
    sink.flush()
    assert producer.records == [("posts_live", b"abc", {"post_id": "abc", "text": "hi"})]
    assert tracker.sent == 1 and tracker.checkpoint() == T0


# --- the read loop against a fake Jetstream ---------------------------------------------------


class FakeSocket:
    def __init__(self, frames, drop_after):
        self.frames, self.drop_after = list(frames), drop_after

    async def recv(self):
        if self.frames:
            return self.frames.pop(0)
        if self.drop_after:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        await asyncio.sleep(3600)  # an idle stream: the duration ends the run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeJetstream:
    """Each connection replays the events at or after the URL's cursor; the first one drops midway."""

    def __init__(self, events, drop_first_after, refuse_first=0):
        self.events, self.drop_first_after, self.refuse = events, drop_first_after, refuse_first
        self.urls, self.opened = [], 0

    def connect(self, url, **kwargs):
        self.urls.append(url)
        if self.refuse:
            self.refuse -= 1
            raise OSError("connection refused")
        self.opened += 1
        found = re.search(r"[?&]cursor=(\d+)", url)
        start = int(found.group(1)) if found else 0
        frames = [json.dumps(e) for e in self.events if e["time_us"] >= start]
        if self.opened == 1 and self.drop_first_after is not None:
            return FakeSocket(frames[: self.drop_first_after], drop_after=True)
        return FakeSocket(frames, drop_after=False)


def stream_events():
    """Ten English posts one second apart, with a like and a French post in between."""
    events = []
    for i in range(10):
        events.append(event(rkey=f"3kfakepost{i:04d}", text=f"Invented post number {i}", time_us=T0 + i * 1_000_000))
    events.insert(3, event(rkey="3kfakelike0001", collection="app.bsky.feed.like", time_us=T0 + 2_500_000))
    events.insert(6, event(rkey="3kfakefr0001", langs=("fr",), time_us=T0 + 4_500_000))
    return events


def run_fake(jetstream, store, duration=0.5, producer=None):
    producer = producer or FakeProducer()
    tracker = DeliveryTracker()
    counts = asyncio.run(
        run(
            SOURCE_CFG,
            LIVE_SOURCES,
            PostClassifier(LIVE_SOURCES),
            KafkaSink(producer, "posts_live", tracker),
            tracker,
            store,
            duration_seconds=duration,
            connect=jetstream.connect,
            rng=random.Random(0),
        )
    )
    return counts, producer


def test_fresh_start_reads_from_now_and_saves_the_cursor(tmp_path):
    store = CursorStore(str(tmp_path / "cursor.json"))
    jetstream = FakeJetstream(stream_events(), drop_first_after=None)
    counts, producer = run_fake(jetstream, store)
    assert "cursor=" not in jetstream.urls[0]
    assert counts["cursor_on_start"] is None
    assert counts["sent"] == 10 and counts["failed"] == 0
    assert store.load() == T0 + 9_000_000
    assert all(topic == "posts_live" for topic, _, _ in producer.records)


def test_dropped_connection_resumes_from_cursor_minus_5_seconds(tmp_path):
    store = CursorStore(str(tmp_path / "cursor.json"))
    jetstream = FakeJetstream(stream_events(), drop_first_after=9)  # 7 English posts, then the drop
    counts, producer = run_fake(jetstream, store)
    last_before_drop = T0 + 6_000_000
    assert counts["reconnects"] == 1
    assert jetstream.urls[1].endswith(f"&cursor={last_before_drop - 5_000_000}")
    assert jetstream.urls[1].split("&cursor=")[0] != jetstream.urls[0]  # moved to the next endpoint
    # every post arrives at least once; the rewound ones repeat with the same post_id
    ids = [message["post_id"] for _, _, message in producer.records]
    assert set(ids) == {post_id(FAKE_DID, f"3kfakepost{i:04d}") for i in range(10)}
    assert len(ids) == 10 + 6  # posts 1..6 were read again after the rewind
    assert store.load() == T0 + 9_000_000


def test_restart_resumes_from_the_saved_cursor(tmp_path):
    store = CursorStore(str(tmp_path / "cursor.json"))
    store.save(T0 + 7_000_000)
    jetstream = FakeJetstream(stream_events(), drop_first_after=None)
    counts, producer = run_fake(jetstream, store)
    assert jetstream.urls[0].endswith(f"&cursor={T0 + 2_000_000}")
    assert counts["cursor_on_start"] == T0 + 7_000_000
    assert [m["post_id"] for _, _, m in producer.records] == [post_id(FAKE_DID, f"3kfakepost{i:04d}") for i in range(2, 10)]


def test_corrupt_cursor_file_starts_from_now(tmp_path):
    path = tmp_path / "cursor.json"
    path.write_text("{half a file", encoding="utf-8")
    jetstream = FakeJetstream(stream_events(), drop_first_after=None)
    counts, _ = run_fake(jetstream, CursorStore(str(path)))
    assert "cursor=" not in jetstream.urls[0]
    assert counts["sent"] == 10
    assert CursorStore(str(path)).load() == T0 + 9_000_000


def test_refused_endpoints_are_tried_in_turn(tmp_path):
    jetstream = FakeJetstream(stream_events(), drop_first_after=None, refuse_first=2)
    counts, _ = run_fake(jetstream, CursorStore(str(tmp_path / "cursor.json")))
    endpoints = LIVE_SOURCES["bluesky"]["endpoints"]
    assert jetstream.urls[:3] == endpoints[:3]
    assert counts["failed_connection_attempts"] == 2 and counts["connections"] == 1
    assert counts["sent"] == 10


class RecordingStore(CursorStore):
    def __init__(self, path):
        super().__init__(path)
        self.saves = []

    def save(self, time_us):
        self.saves.append(time_us)
        super().save(time_us)


def test_rejected_send_holds_the_cursor_back(tmp_path):
    store = RecordingStore(str(tmp_path / "cursor.json"))
    rejected = post_id(FAKE_DID, "3kfakepost0004")
    jetstream = FakeJetstream(stream_events(), drop_first_after=None)
    counts, _ = run_fake(jetstream, store, producer=FakeProducer(reject_post_ids={rejected}))
    assert counts["failed"] == 1 and counts["sent"] == 9
    # the cursor is saved after every event; the one after the rejected post stops just before it
    assert T0 + 4_000_000 - 1 in store.saves
    assert store.saves == sorted(store.saves)


def test_messages_sent_to_kafka_have_no_did_or_handle(tmp_path):
    events = [event(rkey="3kfakepost0001", text=f"@{FAKE_HANDLE} my flight was delayed", time_us=T0)]
    jetstream = FakeJetstream(events, drop_first_after=None)
    _, producer = run_fake(jetstream, CursorStore(str(tmp_path / "cursor.json")))
    (_, key, message), = producer.records
    blob = key.decode("utf-8") + json.dumps(message)
    assert "did:" not in blob and FAKE_HANDLE not in blob
    assert set(message) == set(MESSAGE_FIELDS)


# --- command line -----------------------------------------------------------------------------


def test_producer_options():
    assert parse_args([]).source == "replay"
    args = parse_args(["--source", "bluesky", "--duration-minutes", "15"])
    assert args.source == "bluesky" and args.duration_minutes == 15
    with pytest.raises(SystemExit):
        parse_args(["--duration-minutes", "5"])  # replay has no duration
    with pytest.raises(SystemExit):
        parse_args(["--source", "bluesky", "--duration-minutes", "0"])


def test_live_settings_are_separate_from_replay():
    from src.ml.data import load_config
    from src.stream.settings import mode_settings

    cfg = load_config()
    replay, live = mode_settings(cfg, "replay", env={}), mode_settings(cfg, "live", env={})
    assert live["kafka"]["topic"] == "posts_live" != replay["kafka"]["topic"]
    assert (live["mongodb"]["predictions_collection"], live["mongodb"]["windows_collection"]) == (
        "live_predictions",
        "live_windows",
    )
    assert (live["window"], live["watermark"]) == ("1 minute", "2 minutes")
    assert live["checkpoint_dir"] != replay["checkpoint_dir"]
    env = {"KAFKA_TOPIC": "other", "LIVE_CURSOR_PATH": "/state/c.json"}
    overridden = mode_settings(cfg, "live", env=env)
    assert overridden["kafka"]["topic"] == "other" and overridden["source"]["cursor_path"] == "/state/c.json"
    assert "source" not in mode_settings(cfg, "replay", env=env)  # a live-only key is not added to replay
    with pytest.raises(ValueError):
        mode_settings(cfg, "nonsense", env={})
