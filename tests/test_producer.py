import json
from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from src.ml.data import load_config, prepare
from src.stream.producer import (
    TweetMessage,
    build_message,
    parse_args,
    replay,
    send_offsets,
    select_test_tweets,
    serialize,
)
from src.stream.settings import stream_settings
from tests.conftest import TEST_START

VALID = {
    "tweet_id": "570000000000000001",
    "airline": "United",
    "text": "@united thanks for the upgrade",
    "created_at": "2015-02-23T09:00:00-08:00",
    "produced_at_ms": 1_700_000_000_000,
}


class FakeFuture:
    def __init__(self, error=None):
        self.error = error

    def add_callback(self, fn):
        if self.error is None:
            fn(None)
        return self

    def add_errback(self, fn):
        if self.error is not None:
            fn(self.error)
        return self


class FakeProducer:
    def __init__(self, fail_on=()):
        self.records, self.fail_on, self.flushed = [], set(fail_on), False

    def send(self, topic, key, value):
        self.records.append((topic, key, value))
        return FakeFuture(RuntimeError("boom") if len(self.records) in self.fail_on else None)

    def flush(self):
        self.flushed = True


def first_row(raw, cfg):
    return next(select_test_tweets(raw, cfg).itertuples(index=False))


def test_replayed_tweets_are_exactly_the_test_split_in_time_order(raw, cfg):
    _, expected, _ = prepare(raw, cfg)
    tweets = select_test_tweets(raw, cfg)
    assert sorted(tweets["tweet_id"]) == sorted(expected["tweet_id"])
    assert tweets["created_at"].is_monotonic_increasing
    assert (tweets["created_at"] >= pd.Timestamp(TEST_START)).all()
    assert len(select_test_tweets(raw, cfg, limit=5)) == 5


def test_serialized_message_has_the_agreed_fields_and_no_label(raw, cfg):
    row = first_row(raw, cfg)
    key, value = serialize(build_message(row, run_id=None, produced_at_ms=1_700_000_000_000))
    payload = json.loads(value.decode("utf-8"))

    assert key == row.tweet_id.encode("utf-8")
    assert set(payload) == {"tweet_id", "airline", "text", "created_at", "produced_at_ms"}
    assert payload["tweet_id"] == row.tweet_id
    assert payload["produced_at_ms"] == 1_700_000_000_000
    created_at = datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
    assert created_at.tzinfo is not None
    assert created_at == row.created_at


def test_run_id_is_appended_to_tweet_id(raw, cfg):
    row = first_row(raw, cfg)
    message = build_message(row, run_id="r3", produced_at_ms=1)
    assert message.tweet_id == f"{row.tweet_id}-r3"
    assert serialize(message)[0] == f"{row.tweet_id}-r3".encode("utf-8")


@pytest.mark.parametrize(
    "change",
    [
        {"text": ""},
        {"tweet_id": ""},
        {"airline": ""},
        {"created_at": "2015-02-23T09:00:00"},  # no UTC offset
        {"produced_at_ms": 0},
        {"label": "negative"},  # labels must never be sent
    ],
)
def test_invalid_messages_are_rejected(change):
    TweetMessage(**VALID)
    with pytest.raises(ValidationError):
        TweetMessage(**{**VALID, **change})


def test_send_offsets_scale_event_time_gaps():
    created_at = pd.Series(
        pd.to_datetime(["2015-02-23T08:00:00Z", "2015-02-23T08:10:00Z", "2015-02-23T09:00:00Z"])
    )
    assert send_offsets(created_at, speedup=600) == [0.0, 1.0, 6.0]


def test_replay_counts_successes_and_failures_and_flushes(raw, cfg):
    tweets = select_test_tweets(raw, cfg, limit=6)
    fake = FakeProducer(fail_on={2})
    sent, failed = replay(fake, "tweets", tweets, speedup=None, run_id="x")
    assert (sent, failed) == (5, 1)
    assert fake.flushed
    assert all(topic == "tweets" for topic, _, _ in fake.records)
    assert all(json.loads(value)["tweet_id"].endswith("-x") for _, _, value in fake.records)


def test_speedup_and_max_rate_are_mutually_exclusive():
    assert parse_args(["--max-rate", "--limit", "500"]).max_rate
    assert parse_args(["--speedup", "600"]).speedup == 600
    with pytest.raises(SystemExit):
        parse_args(["--speedup", "600", "--max-rate"])


def test_environment_overrides_stream_settings():
    cfg = {"stream": {"kafka": {"bootstrap_servers": "localhost:29092", "topic": "tweets"}}}
    assert stream_settings(cfg, env={})["kafka"]["bootstrap_servers"] == "localhost:29092"
    overridden = stream_settings(cfg, env={"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"})
    assert overridden["kafka"] == {"bootstrap_servers": "kafka:9092", "topic": "tweets"}
    assert cfg["stream"]["kafka"]["bootstrap_servers"] == "localhost:29092"


def test_stream_settings_cover_every_configurable_value():
    settings = stream_settings(
        load_config(),
        env={
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "MONGODB_URI": "mongodb://mongodb:27017",
            "STREAM_TRIGGER": "1 second",
            "SPARK_MASTER": "local[4]",
        },
    )
    assert settings["kafka"] == {"bootstrap_servers": "kafka:9092", "topic": "tweets"}
    assert settings["mongodb"]["uri"] == "mongodb://mongodb:27017"
    assert settings["mongodb"]["predictions_collection"] == "predictions"
    assert settings["mongodb"]["windows_collection"] == "airline_windows"
    assert (settings["window"], settings["watermark"], settings["trigger"]) == (
        "1 hour",
        "30 minutes",
        "1 second",
    )
    assert settings["spark"]["master"] == "local[4]"
    assert settings["spark"]["driver_memory"] == "1g"
