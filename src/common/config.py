import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "social_media_stream"

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"
MONGO_RESULTS_URI = f"{MONGO_URI}/{DB_NAME}.{COLLECTION_NAME}"

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
CSV_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Tweets.csv")
PIPELINE_CONFIG_PATH = os.path.join(BASE_DIR, "config", "pipeline.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sentiment_analysis")
