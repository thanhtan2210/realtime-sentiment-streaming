# Image for the Spark Structured Streaming processor.
# Build from the repo root:
#   docker build -f docker/spark.Dockerfile -t realtime-sentiment-spark .
#
# The Kafka and MongoDB connectors (and their dependencies) are downloaded
# here, at build time, and listed in spark.jars. A container started from
# this image needs no network access to fetch jars.
#
# bookworm, not the default python:3.11-slim (Debian 13), because Debian 13
# no longer ships openjdk-17 and Spark 3.5 does not support Java 21.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYSPARK_PYTHON=python3 \
    SPARK_CONF_DIR=/opt/spark/conf

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-spark.txt ./
RUN pip install -r requirements-spark.txt

# SPARK_KAFKA_VERSION must equal the pyspark version in requirements-spark.txt;
# fetch_spark_jars.py fails the build if it does not.
ARG SPARK_KAFKA_VERSION=3.5.9
ARG MONGO_CONNECTOR_VERSION=10.7.0
COPY docker/fetch_spark_jars.py /tmp/fetch_spark_jars.py
RUN python /tmp/fetch_spark_jars.py \
      "org.apache.spark:spark-sql-kafka-0-10_2.12:${SPARK_KAFKA_VERSION}" \
      "org.mongodb.spark:mongo-spark-connector_2.12:${MONGO_CONNECTOR_VERSION}" \
 && rm /tmp/fetch_spark_jars.py

# The processor runs Spark in local mode, so the driver only talks to itself.
# Without this, a container whose hostname does not resolve (for example
# `docker run --network none`) cannot start a session.
ENV SPARK_LOCAL_IP=127.0.0.1     SPARK_LOCAL_HOSTNAME=localhost     PYTHONPATH=/app

COPY src/ src/
COPY config/ config/
COPY models/ models/
