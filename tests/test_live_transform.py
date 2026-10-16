"""Spark logic of the live mode, on small batch DataFrames (needs Java). All posts are invented."""
import json
import sys
from datetime import datetime, timezone

import pytest

pytest.importorskip("pyspark", reason="install requirements-spark.txt to run the Spark tests")

from src.stream.sources.bluesky import PostClassifier, to_message  # noqa: E402
from src.stream.sources.filters import load_live_config  # noqa: E402
from src.stream.transform import (  # noqa: E402
    ALL_TOPICS,
    LABELS,
    drop_repeated_posts,
    live_prediction_documents,
    live_window_counts,
    live_window_documents,
    parse_live_messages,
    with_sentiment,
)

pytestmark = pytest.mark.spark

KAFKA_TIME = datetime(2027, 1, 15, 8, 0, 30)
PREDICTION_SCHEMA = "post_id string, event_time timestamp, topic string, sentiment string"


def utc(minute, second=0):
    return datetime(2027, 1, 15, 8, minute, second, tzinfo=timezone.utc)


def as_utc(collected: datetime) -> datetime:
    return collected.astimezone(timezone.utc)


def fake_event(i, text, second):
    return {
        "did": "did:plc:testauthor0000000000000",
        "time_us": int(utc(0, second).timestamp() * 1_000_000),
        "kind": "commit",
        "commit": {
            "operation": "create",
            "collection": "app.bsky.feed.post",
            "rkey": f"3kfakepost{i:04d}",
            "record": {"text": text, "langs": ["en"], "createdAt": "2027-01-15T08:00:00.000Z"},
        },
    }


@pytest.fixture
def kafka_records(spark):
    """What Kafka would deliver: three invented posts (one sent twice) and two bad records."""
    classifier = PostClassifier(load_live_config())
    texts = ["united airlines lost my bag again", "my flight was delayed", "what a lovely morning"]
    messages = []
    for i, text in enumerate(texts):
        message, _ = to_message(fake_event(i, text, second=10 * i), classifier, received_at_ms=1_800_000_000_000)
        message["produced_at_ms"] = 1_800_000_000_100
        messages.append(message)
    values = [json.dumps(m).encode("utf-8") for m in messages + messages[:1]]
    values += [b"not json", json.dumps({**messages[1], "post_id": None}).encode("utf-8")]
    records = spark.createDataFrame([(bytearray(v), KAFKA_TIME) for v in values], "value binary, timestamp timestamp")
    return messages, records


def test_parse_types_the_event_time_and_drops_bad_records(kafka_records):
    messages, records = kafka_records
    rows = parse_live_messages(records).collect()
    assert len(rows) == 4  # three posts, one of them twice
    first = next(r for r in rows if r["post_id"] == messages[0]["post_id"])
    assert first["topic"] == "airline:United"
    assert as_utc(first["event_time"]) == utc(0, 0)
    assert first["received_at_ms"] == 1_800_000_000_000 and first["produced_at_ms"] == 1_800_000_000_100


def test_live_prediction_documents_are_one_per_post(kafka_records):
    messages, records = kafka_records
    posts = drop_repeated_posts(parse_live_messages(records), "2 minutes")
    documents = live_prediction_documents(with_sentiment(posts))
    assert documents.columns == ["_id", "topic", "text", "sentiment", "event_time", "received_at_ms", "produced_at_ms"]
    rows = documents.collect()
    assert sorted(r["_id"] for r in rows) == sorted(m["post_id"] for m in messages)
    assert {r["sentiment"] for r in rows} <= set(LABELS)
    assert {r["topic"] for r in rows} == {"airline:United", "air_travel", "general"}


def test_live_windows_count_each_topic_and_all(spark):
    predictions = spark.createDataFrame(
        [
            ("a", utc(0, 5), "airline:United", "negative"),
            ("b", utc(0, 20), "airline:United", "positive"),
            ("c", utc(0, 40), "air_travel", "negative"),
            ("d", utc(0, 59), "general", "neutral"),
            ("e", utc(1, 0), "general", "positive"),  # next minute
            ("a", utc(0, 5), "airline:United", "negative"),  # post "a" delivered again
        ],
        PREDICTION_SCHEMA,
    )
    counts = live_window_counts(drop_repeated_posts(predictions, "2 minutes"), "1 minute")
    rows = {r["_id"]: r.asDict() for r in live_window_documents(counts).collect()}

    assert set(rows) == {
        "airline:United|2027-01-15T08:00:00Z",
        "air_travel|2027-01-15T08:00:00Z",
        "general|2027-01-15T08:00:00Z",
        "all|2027-01-15T08:00:00Z",
        "general|2027-01-15T08:01:00Z",
        "all|2027-01-15T08:01:00Z",
    }
    united = rows["airline:United|2027-01-15T08:00:00Z"]
    assert (united["negative"], united["neutral"], united["positive"], united["total"]) == (1, 0, 1, 2)
    assert united["negative_share"] == pytest.approx(0.5)
    everything = rows[f"{ALL_TOPICS}|2027-01-15T08:00:00Z"]
    assert (everything["negative"], everything["neutral"], everything["positive"], everything["total"]) == (2, 1, 1, 4)
    assert everything["negative_share"] == pytest.approx(0.5)
    assert as_utc(everything["window_start"]) == utc(0) and as_utc(everything["window_end"]) == utc(1)
    assert rows["all|2027-01-15T08:01:00Z"]["total"] == 1

    # every post is in exactly one topic row and in the `all` row of its window
    for start in ("2027-01-15T08:00:00Z", "2027-01-15T08:01:00Z"):
        topics = sum(r["total"] for key, r in rows.items() if key.endswith(start) and r["topic"] != ALL_TOPICS)
        assert topics == rows[f"all|{start}"]["total"]


def test_live_window_totals_match_the_predictions(kafka_records):
    messages, records = kafka_records
    predictions = with_sentiment(drop_repeated_posts(parse_live_messages(records), "2 minutes"))
    counts = live_window_counts(predictions, "1 minute").collect()
    assert sum(r["total"] for r in counts if r["topic"] == ALL_TOPICS) == len(messages)
    assert sum(r["total"] for r in counts if r["topic"] != ALL_TOPICS) == len(messages)
    for r in counts:
        assert r["negative"] + r["neutral"] + r["positive"] == r["total"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Spark's streaming state store needs Hadoop's native Windows libraries",
)
def test_streaming_live_query_counts_a_reread_post_once(spark, tmp_path):
    """The real streaming operators (dedupe, then window per topic and `all`) across two micro-batches."""
    source = str(tmp_path / "in")
    first = [("a", utc(0, 5), "air_travel", "negative")]
    second = [("a", utc(0, 5), "air_travel", "negative"), ("b", utc(0, 20), "general", "positive")]
    for rows in (first, second):
        spark.createDataFrame(rows, PREDICTION_SCHEMA).coalesce(1).write.mode("append").parquet(source)

    updates = []
    stream = spark.readStream.schema(PREDICTION_SCHEMA).option("maxFilesPerTrigger", 1).parquet(source)
    query = (
        live_window_documents(live_window_counts(drop_repeated_posts(stream, "2 minutes"), "1 minute"))
        .writeStream.outputMode("update")
        .foreachBatch(lambda batch, _: updates.extend(row.asDict() for row in batch.collect()))
        .option("checkpointLocation", str(tmp_path / "checkpoint"))
        .trigger(availableNow=True)
        .start()
    )
    assert query.awaitTermination(180)
    assert query.exception() is None

    latest = {row["_id"]: row for row in updates}
    assert latest["air_travel|2027-01-15T08:00:00Z"]["total"] == 1
    assert latest["general|2027-01-15T08:00:00Z"]["total"] == 1
    assert latest["all|2027-01-15T08:00:00Z"]["total"] == 2
