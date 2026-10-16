"""Spark transformations shared by the streaming job, the tests and the benchmark.

Everything here takes and returns DataFrames, so it works the same on a
streaming DataFrame and on a small batch DataFrame in a test. The functions
named `live_*` (and `drop_repeated_posts`) are for the live mode's Bluesky
posts; the others for the replayed tweets.
"""
import os
import sys

import joblib
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import LongType, StringType, StructField, StructType

from src.common import config

LABELS = ["negative", "neutral", "positive"]

# The JSON value written by src.stream.producer.
MESSAGE_SCHEMA = StructType(
    [
        StructField("tweet_id", StringType()),
        StructField("airline", StringType()),
        StructField("text", StringType()),
        StructField("created_at", StringType()),
        StructField("produced_at_ms", LongType()),
    ]
)

# The JSON value written by src.stream.sources.bluesky (live mode).
LIVE_MESSAGE_SCHEMA = StructType(
    [
        StructField("post_id", StringType()),
        StructField("text", StringType()),
        StructField("topic", StringType()),
        StructField("event_time", StringType()),
        StructField("created_at", StringType()),
        StructField("received_at_ms", LongType()),
        StructField("produced_at_ms", LongType()),
    ]
)
# Live windows also hold one row per window over every topic.
ALL_TOPICS = "all"

_model = None


def get_model():
    """Load the model once per Python worker process and keep it for later batches."""
    global _model
    if _model is None:
        _model = joblib.load(config.MODEL_PATH)
        print(f"Loaded sentiment model in worker pid {os.getpid()}", file=sys.stderr, flush=True)
    return _model


def _predict(text: pd.Series) -> pd.Series:
    if text.empty:
        return pd.Series([], dtype="object")
    return pd.Series(get_model().predict(text.fillna("").tolist()), index=text.index)


predict_sentiment = pandas_udf(_predict, StringType())


def parse_messages(kafka_df: DataFrame) -> DataFrame:
    """Kafka records (`value`, `timestamp`) -> one typed row per valid tweet."""
    return (
        kafka_df.select(
            F.from_json(F.col("value").cast("string"), MESSAGE_SCHEMA).alias("m"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("m.*", "kafka_timestamp")
        .withColumn("event_time", F.to_timestamp("created_at"))
        .filter(
            F.col("tweet_id").isNotNull()
            & F.col("airline").isNotNull()
            & (F.length(F.trim(F.col("text"))) > 0)
            & F.col("event_time").isNotNull()
        )
    )


def with_sentiment(tweets: DataFrame) -> DataFrame:
    return tweets.withColumn("sentiment", predict_sentiment(F.col("text")))


def prediction_documents(predictions: DataFrame) -> DataFrame:
    """One MongoDB document per tweet; `_id` is the tweet id, so a re-run overwrites."""
    return predictions.select(
        F.col("tweet_id").alias("_id"),
        "airline",
        "text",
        F.col("event_time").alias("created_at"),
        "sentiment",
        "produced_at_ms",
        "kafka_timestamp",
    )


def label_counts() -> list:
    return [F.sum(F.when(F.col("sentiment") == label, 1).otherwise(0)).alias(label) for label in LABELS]


def window_counts(predictions: DataFrame, window: str, watermark: str) -> DataFrame:
    """Tweets per label for each airline and event-time window.

    A tweet delivered more than once within the watermark is counted once.
    """
    counts = label_counts()
    tweets = predictions.withWatermark("event_time", watermark)
    if predictions.isStreaming:
        tweets = tweets.dropDuplicatesWithinWatermark(["tweet_id"])
    else:
        # Spark rejects dropDuplicatesWithinWatermark on batch DataFrames;
        # without a watermark the plain version does the same job.
        tweets = tweets.dropDuplicates(["tweet_id"])
    return (
        tweets.groupBy(F.window("event_time", window).alias("window"), "airline")
        .agg(F.count(F.lit(1)).alias("total"), *counts)
        .withColumn("negative_share", F.col("negative") / F.col("total"))
    )


def window_documents(counts: DataFrame) -> DataFrame:
    """One MongoDB document per airline and window; `_id` is `airline|window_start` (UTC)."""
    window_start = F.date_format(F.col("window.start"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
    return counts.select(
        F.concat_ws("|", F.col("airline"), window_start).alias("_id"),
        "airline",
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        *LABELS,
        "total",
        "negative_share",
    )


# --- live mode ------------------------------------------------------------------------------


def parse_live_messages(kafka_df: DataFrame) -> DataFrame:
    """Kafka records (`value`, `timestamp`) -> one typed row per valid live post.

    `event_time` is the Jetstream time of the post, as sent by the producer.
    """
    return (
        kafka_df.select(
            F.from_json(F.col("value").cast("string"), LIVE_MESSAGE_SCHEMA).alias("m"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("m.*", "kafka_timestamp")
        .withColumn("event_time", F.to_timestamp("event_time"))
        .filter(
            F.col("post_id").isNotNull()
            & F.col("topic").isNotNull()
            & (F.length(F.trim(F.col("text"))) > 0)
            & F.col("event_time").isNotNull()
        )
    )


def drop_repeated_posts(posts: DataFrame, watermark: str) -> DataFrame:
    """Keep the first delivery of each post_id (the producer re-reads a few seconds after a reconnect).

    On a stream, a post_id is remembered for `watermark` of event time, and posts
    older than the watermark are dropped as late.
    """
    posts = posts.withWatermark("event_time", watermark)
    if posts.isStreaming:
        return posts.dropDuplicatesWithinWatermark(["post_id"])
    return posts.dropDuplicates(["post_id"])


def live_prediction_documents(predictions: DataFrame) -> DataFrame:
    """One MongoDB document per post; `_id` is post_id. `written_at` is added by the writer.

    The three times for the latencies: `created_at` (the post's own createdAt),
    `received_at_ms` (when the producer read it) and `written_at`.
    """
    return predictions.select(
        F.col("post_id").alias("_id"),
        "topic",
        "text",
        "sentiment",
        F.to_timestamp("created_at").alias("created_at"),
        "event_time",
        "received_at_ms",
        "produced_at_ms",
    )


def live_window_counts(predictions: DataFrame, window: str) -> DataFrame:
    """Posts per label for each topic and event-time window, plus a row for all topics together.

    Takes the output of `drop_repeated_posts` (with sentiment), so each post is
    counted once in its own topic and once in `all`.
    """
    rows = predictions.withColumn("topic", F.explode(F.array(F.col("topic"), F.lit(ALL_TOPICS))))
    return (
        rows.groupBy(F.window("event_time", window).alias("window"), "topic")
        .agg(F.count(F.lit(1)).alias("total"), *label_counts())
        .withColumn("negative_share", F.col("negative") / F.col("total"))
    )


def live_window_documents(counts: DataFrame) -> DataFrame:
    """One MongoDB document per topic and window; `_id` is `topic|window_start` (UTC)."""
    window_start = F.date_format(F.col("window.start"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
    return counts.select(
        F.concat_ws("|", F.col("topic"), window_start).alias("_id"),
        "topic",
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        *LABELS,
        "total",
        "negative_share",
    )
