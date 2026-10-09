import os
import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

from src.common import config, logger as common_logger

logger = common_logger.get_logger(__name__)

# Schema definition for Kafka data
schema = StructType([
    StructField("text", StringType()),
    StructField("user", StringType()),
    StructField("platform", StringType()),
    StructField("timestamp", StringType()),
    StructField("country", StringType())
])

def main():
    # Initialize Spark Session
    logger.info("Starting Spark Session with Kafka & MongoDB connectors...")
    spark = SparkSession.builder \
        .appName("SentimentAnalysisStreaming") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.mongodb.output.uri", config.MONGO_RESULTS_URI) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Load NLP model
    if os.path.exists(config.MODEL_PATH):
        logger.info(f"Loading pre-trained model from {config.MODEL_PATH}...")
        model = joblib.load(config.MODEL_PATH)
    else:
        logger.error(f"Model not found at {config.MODEL_PATH}. Please run train.py first.")
        return

    # Pandas UDF
    @pandas_udf(StringType())
    def predict_sentiment_pandas_udf(text_series: pd.Series) -> pd.Series:
        clean_series = text_series.fillna("")
        predictions = model.predict(clean_series.tolist())
        return pd.Series(predictions)

    # Read data stream
    logger.info(f"Subscribing to Kafka topic: {config.TOPIC_NAME}...")
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", config.TOPIC_NAME) \
        .option("startingOffsets", "latest") \
        .load()

    json_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Data Cleaning
    clean_df = json_df.filter((col("text").isNotNull()) & (col("text") != ""))

    results_df = clean_df.withColumn("sentiment", predict_sentiment_pandas_udf(col("text")))

    # Write to MongoDB
    logger.info(f"Initializing streaming query to MongoDB: {config.MONGO_RESULTS_URI}")
    query = results_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: batch_df.write \
            .format("mongo") \
            .mode("append") \
            .save()) \
        .option("checkpointLocation", config.CHECKPOINT_PATH) \
        .start()

    logger.info(f"Spark Streaming is active and processing '{config.TOPIC_NAME}'...")
    query.awaitTermination()

if __name__ == "__main__":
    main()
