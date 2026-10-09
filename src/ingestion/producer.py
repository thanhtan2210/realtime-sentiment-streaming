import json
import time
import pandas as pd
import os
from kafka import KafkaProducer
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

from src.common import config, logger as common_logger

logger = common_logger.get_logger(__name__)

# Data Schema Definition
class SentimentData(BaseModel):
    text: str = Field(..., min_length=1)
    user: str = Field(default="Unknown")
    platform: str = Field(default="Unknown")
    timestamp: Optional[str] = None
    country: str = Field(default="Unknown")

def get_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        return None

def main():
    if not os.path.exists(config.CSV_DATA_PATH):
        logger.error(f"File not found: {config.CSV_DATA_PATH}")
        return

    # Load data
    df = pd.read_csv(config.CSV_DATA_PATH)
    logger.info(f"Successfully loaded {len(df)} records from CSV.")

    producer = get_producer()
    if not producer:
        return

    logger.info(f"Starting to stream data to Kafka topic: {config.TOPIC_NAME}...")

    success_count = 0
    error_count = 0

    for _, row in df.iterrows():
        try:
            raw_data = {
                "text": str(row.get("Text", "")).strip(),
                "user": str(row.get("User", "")).strip(),
                "platform": str(row.get("Platform", "")).strip(),
                "timestamp": str(row.get("Timestamp", "")),
                "country": str(row.get("Country", "")).strip()
            }
            
            validated_data = SentimentData(**raw_data)
            producer.send(config.TOPIC_NAME, value=validated_data.model_dump())
            success_count += 1
            
            if success_count % 50 == 0:
                logger.info(f"Sent {success_count} valid records...")
            
        except ValidationError as ve:
            error_count += 1
            logger.warning(f"Invalid record skipped: {ve.json()}")
        except Exception as e:
            logger.error(f"Error sending record: {e}")

        time.sleep(0.5)

    producer.flush()
    logger.info(f"Streaming finished. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
