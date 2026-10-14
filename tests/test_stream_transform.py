"""Spark logic of the streaming job, run on small batch DataFrames (needs Java)."""
import sys
from datetime import datetime, timezone

import pytest

pytest.importorskip("pyspark", reason="install requirements-spark.txt to run the Spark tests")

from src.stream.producer import build_message, select_test_tweets, serialize  # noqa: E402
from src.stream.transform import (  # noqa: E402
    LABELS,
    parse_messages,
    prediction_documents,
    window_counts,
    window_documents,
    with_sentiment,
)

pytestmark = pytest.mark.spark

KAFKA_TIME = datetime(2026, 1, 1, 12, 0, 0)
PREDICTION_SCHEMA = "tweet_id string, event_time timestamp, airline string, sentiment string"


def utc(day, hour, minute=0):
    return datetime(2015, 2, day, hour, minute, tzinfo=timezone.utc)


def as_utc(collected: datetime) -> datetime:
    """Spark hands timestamps back as naive datetimes in the machine's local time zone."""
    return collected.astimezone(timezone.utc)


@pytest.fixture
def kafka_records(spark, raw, cfg):
    """What the Kafka source would deliver for the fixture's test tweets, plus two bad records."""
    tweets = select_test_tweets(raw, cfg)
    values = [
        serialize(build_message(row, run_id=None, produced_at_ms=1_700_000_000_000 + i))[1]
        for i, row in enumerate(tweets.itertuples(index=False))
    ]
    values += [b"not json", b'{"tweet_id": "1", "airline": "United", "text": " ", "created_at": "x"}']
    records = spark.createDataFrame(
        [(bytearray(v), KAFKA_TIME) for v in values], "value binary, timestamp timestamp"
    )
    return tweets, records


def test_parse_keeps_valid_messages_and_types_the_event_time(kafka_records):
    tweets, records = kafka_records
    rows = parse_messages(records).collect()

    assert len(rows) == len(tweets)  # the two bad records are dropped
    by_id = {row["tweet_id"]: row for row in rows}
    assert set(by_id) == set(tweets["tweet_id"])
    first = tweets.iloc[0]
    parsed = by_id[first["tweet_id"]]
    assert parsed["airline"] == first["airline"]
    assert parsed["text"] == first["text"]
    assert as_utc(parsed["event_time"]) == first["created_at"].to_pydatetime()
    assert parsed["produced_at_ms"] == 1_700_000_000_000
    assert parsed["kafka_timestamp"] is not None


def test_predictions_use_the_three_labels_and_tweet_id_as_document_id(kafka_records):
    tweets, records = kafka_records
    documents = prediction_documents(with_sentiment(parse_messages(records)))

    assert documents.columns == [
        "_id",
        "airline",
        "text",
        "created_at",
        "sentiment",
        "produced_at_ms",
        "kafka_timestamp",
    ]
    rows = documents.collect()
    assert sorted(row["_id"] for row in rows) == sorted(tweets["tweet_id"])
    assert {row["sentiment"] for row in rows} <= set(LABELS)


def test_window_counts_and_negative_share(spark):
    predictions = spark.createDataFrame(
        [
            ("1", utc(23, 8, 5), "United", "negative"),
            ("2", utc(23, 8, 20), "United", "negative"),
            ("3", utc(23, 8, 40), "United", "positive"),
            ("4", utc(23, 8, 59), "United", "neutral"),
            ("5", utc(23, 9, 0), "United", "positive"),  # next window
            ("6", utc(23, 8, 30), "Delta", "neutral"),
        ],
        PREDICTION_SCHEMA,
    )
    documents = window_documents(window_counts(predictions, "1 hour", "30 minutes"))
    rows = {row["_id"]: row.asDict() for row in documents.collect()}

    assert set(rows) == {
        "United|2015-02-23T08:00:00Z",
        "United|2015-02-23T09:00:00Z",
        "Delta|2015-02-23T08:00:00Z",
    }
    united = rows["United|2015-02-23T08:00:00Z"]
    assert (united["negative"], united["neutral"], united["positive"], united["total"]) == (2, 1, 1, 4)
    assert united["negative_share"] == pytest.approx(0.5)
    assert as_utc(united["window_start"]) == utc(23, 8)
    assert as_utc(united["window_end"]) == utc(23, 9)

    next_hour = rows["United|2015-02-23T09:00:00Z"]
    assert (next_hour["negative"], next_hour["total"], next_hour["negative_share"]) == (0, 1, 0.0)
    delta = rows["Delta|2015-02-23T08:00:00Z"]
    assert (delta["neutral"], delta["total"], delta["negative_share"]) == (1, 1, 0.0)


def test_a_tweet_delivered_twice_is_counted_once_in_its_window(spark):
    predictions = spark.createDataFrame(
        [
            ("1", utc(23, 8, 5), "United", "negative"),
            ("1", utc(23, 8, 5), "United", "negative"),  # the same tweet again
            ("2", utc(23, 8, 20), "United", "positive"),
            ("3", utc(23, 8, 20), "Delta", "neutral"),
            ("3", utc(23, 8, 20), "Delta", "neutral"),
            ("3", utc(23, 8, 20), "Delta", "neutral"),
        ],
        PREDICTION_SCHEMA,
    )
    rows = {
        row["_id"]: row.asDict()
        for row in window_documents(window_counts(predictions, "1 hour", "30 minutes")).collect()
    }

    united = rows["United|2015-02-23T08:00:00Z"]
    assert (united["negative"], united["positive"], united["total"]) == (1, 1, 2)
    assert united["negative_share"] == pytest.approx(0.5)
    delta = rows["Delta|2015-02-23T08:00:00Z"]
    assert (delta["neutral"], delta["total"]) == (1, 1)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Spark's streaming state store needs Hadoop's native Windows libraries",
)
def test_streaming_query_counts_a_redelivered_tweet_once(spark, tmp_path):
    """The real streaming operator, across two micro-batches, in update mode."""
    source = str(tmp_path / "in")
    first = [("1", utc(23, 8, 5), "United", "negative")]
    second = [("1", utc(23, 8, 5), "United", "negative"), ("2", utc(23, 8, 20), "United", "positive")]
    for rows in (first, second):
        spark.createDataFrame(rows, PREDICTION_SCHEMA).coalesce(1).write.mode("append").parquet(source)

    updates = []
    stream = spark.readStream.schema(PREDICTION_SCHEMA).option("maxFilesPerTrigger", 1).parquet(source)
    query = (
        window_documents(window_counts(stream, "1 hour", "30 minutes"))
        .writeStream.outputMode("update")
        .foreachBatch(lambda batch, _: updates.extend(row.asDict() for row in batch.collect()))
        .option("checkpointLocation", str(tmp_path / "checkpoint"))
        .trigger(availableNow=True)
        .start()
    )
    assert query.awaitTermination(180)
    assert query.exception() is None

    latest = {row["_id"]: row for row in updates}  # update mode: the last row per key wins
    assert set(latest) == {"United|2015-02-23T08:00:00Z"}
    united = latest["United|2015-02-23T08:00:00Z"]
    assert (united["negative"], united["positive"], united["total"]) == (1, 1, 2)


def test_window_totals_match_the_number_of_predictions(kafka_records):
    tweets, records = kafka_records
    counts = window_counts(with_sentiment(parse_messages(records)), "1 hour", "30 minutes").collect()

    assert sum(row["total"] for row in counts) == len(tweets)
    for row in counts:
        assert row["negative"] + row["neutral"] + row["positive"] == row["total"]
        assert row["negative_share"] == pytest.approx(row["negative"] / row["total"])
