"""Spark Structured Streaming job: Kafka -> sentiment model -> MongoDB.

Usage (inside the Spark image, where the Kafka and MongoDB connectors are installed):
    python -m src.stream.processor

Two queries read the same topic, each with its own checkpoint:
  predictions      one document per tweet, `_id` = tweet_id
  airline_windows  label counts per airline and event-time window, `_id` = airline|window_start

Both sinks replace documents by `_id`, so replaying a batch does not create duplicates.
The process exits with a non-zero code if the model file is missing, Kafka or
MongoDB cannot be reached at start-up, or a query fails.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQueryListener

from src.common import config, logger as common_logger
from src.ml.data import load_config
from src.stream.settings import stream_settings
from src.stream.transform import (
    parse_messages,
    prediction_documents,
    window_counts,
    window_documents,
    with_sentiment,
)

logger = common_logger.get_logger(__name__)


class StartupError(RuntimeError):
    pass


def check_model() -> None:
    if not os.path.exists(config.MODEL_PATH):
        raise StartupError(f"Model not found at {config.MODEL_PATH}; run `python -m src.ml.train`")


def check_kafka(kafka: dict) -> None:
    try:
        consumer = KafkaConsumer(bootstrap_servers=kafka["bootstrap_servers"])
        partitions = consumer.partitions_for_topic(kafka["topic"])
        consumer.close()
    except KafkaError as exc:
        raise StartupError(f"Cannot reach Kafka at {kafka['bootstrap_servers']}: {exc!r}") from exc
    if not partitions:
        raise StartupError(f"Kafka topic '{kafka['topic']}' does not exist")
    logger.info(f"Kafka OK: topic '{kafka['topic']}' has {len(partitions)} partitions")


def check_mongodb(mongodb: dict) -> None:
    try:
        client = MongoClient(mongodb["uri"], serverSelectionTimeoutMS=10_000)
        client.admin.command("ping")
        client.close()
    except PyMongoError as exc:
        raise StartupError(f"Cannot reach MongoDB at {mongodb['uri']}: {exc!r}") from exc
    logger.info("MongoDB OK")


def build_session(spark_cfg: dict) -> SparkSession:
    return (
        SparkSession.builder.appName("sentiment-stream")
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


class ProgressLogger(StreamingQueryListener):
    """Log one `PROGRESS {json}` line per micro-batch that had input (read by the benchmark)."""

    def onQueryStarted(self, event):
        logger.info(f"Query started: {event.name}")

    def onQueryProgress(self, event):
        progress = json.loads(event.progress.json)
        if progress["numInputRows"] > 0:
            keep = ("name", "batchId", "timestamp", "numInputRows", "processedRowsPerSecond", "durationMs")
            line = {key: progress.get(key) for key in keep}
            line["rowsDroppedByWatermark"] = sum(
                op.get("numRowsDroppedByWatermark", 0) for op in progress.get("stateOperators", [])
            )
            logger.info("PROGRESS " + json.dumps(line))

    def onQueryIdle(self, event):
        pass

    def onQueryTerminated(self, event):
        logger.info(f"Query terminated: {event.id} exception={event.exception}")


def mongo_writer(mongodb: dict, collection: str):
    def write(documents) -> None:
        (
            documents.write.format("mongodb")
            .mode("append")
            .option("connection.uri", mongodb["uri"])
            .option("database", mongodb["database"])
            .option("collection", collection)
            .option("operationType", "replace")
            .option("idFieldList", "_id")
            .option("upsertDocument", "true")
            .save()
        )

    return write


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def start_queries(spark: SparkSession, settings: dict) -> list:
    kafka, mongodb = settings["kafka"], settings["mongodb"]
    checkpoint_dir = os.path.join(config.BASE_DIR, settings["checkpoint_dir"])
    write_predictions = mongo_writer(mongodb, mongodb["predictions_collection"])
    write_windows = mongo_writer(mongodb, mongodb["windows_collection"])

    source = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka["bootstrap_servers"])
        .option("subscribe", kafka["topic"])
        .option("startingOffsets", "earliest")
        .load()
    )
    predictions = with_sentiment(parse_messages(source))

    def predictions_batch(batch_df, batch_id: int) -> None:
        # Run the model first, so `written_at` is stamped after inference and
        # just before the MongoDB write starts.
        documents = prediction_documents(batch_df).persist()
        started = time.monotonic()
        rows = documents.count()
        predicted = time.monotonic()
        written_at = utc_now()
        write_predictions(documents.withColumn("written_at", F.lit(written_at)))
        written = time.monotonic()
        documents.unpersist()
        if rows:
            timing = {
                "sink": "predictions",
                "batchId": batch_id,
                "rows": rows,
                "writtenAt": written_at.isoformat(timespec="milliseconds"),
                "readAndPredictMs": round((predicted - started) * 1000),
                "mongoWriteMs": round((written - predicted) * 1000),
            }
            logger.info("WRITE " + json.dumps(timing))

    def windows_batch(batch_df, batch_id: int) -> None:
        documents = window_documents(batch_df).persist()
        documents.count()  # aggregate first, so `updated_at` is the time of the write
        write_windows(documents.withColumn("updated_at", F.lit(utc_now())))
        documents.unpersist()

    trigger = settings["trigger"]
    return [
        predictions.writeStream.queryName("predictions")
        .foreachBatch(predictions_batch)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "predictions"))
        .trigger(processingTime=trigger)
        .start(),
        window_counts(predictions, settings["window"], settings["watermark"])
        .writeStream.queryName("airline_windows")
        .outputMode("update")
        .foreachBatch(windows_batch)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "airline_windows"))
        .trigger(processingTime=trigger)
        .start(),
    ]


def main() -> int:
    settings = stream_settings(load_config())
    try:
        check_model()
        check_kafka(settings["kafka"])
        check_mongodb(settings["mongodb"])
    except StartupError as exc:
        logger.error(str(exc))
        return 1

    spark = build_session(settings["spark"])
    spark.sparkContext.setLogLevel("WARN")
    spark.streams.addListener(ProgressLogger())
    logger.info(
        f"Starting queries: trigger={settings['trigger']}, window={settings['window']}, "
        f"watermark={settings['watermark']}, master={settings['spark']['master']}"
    )
    try:
        start_queries(spark, settings)
        spark.streams.awaitAnyTermination()
    except Exception as exc:  # a failed query surfaces here as StreamingQueryException
        logger.error(f"Streaming query failed: {exc}")
        return 1
    finally:
        spark.stop()
    # Queries only stop on failure or shutdown; a clean return here is unexpected.
    logger.error("A streaming query stopped")
    return 1


if __name__ == "__main__":
    sys.exit(main())
