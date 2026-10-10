"""Spark transformations shared by the streaming job, the tests and the benchmark.

Everything here takes and returns DataFrames, so it works the same on a
streaming DataFrame and on a small batch DataFrame in a test.
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


def window_counts(predictions: DataFrame, window: str, watermark: str) -> DataFrame:
    """Tweets per label for each airline and event-time window.

    A tweet delivered more than once within the watermark is counted once.
    """
    counts = [
        F.sum(F.when(F.col("sentiment") == label, 1).otherwise(0)).alias(label) for label in LABELS
    ]
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
