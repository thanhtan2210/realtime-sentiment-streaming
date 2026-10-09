import os

# Paths. Kafka, MongoDB and Spark settings live in config/pipeline.yaml.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
CSV_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Tweets.csv")
PIPELINE_CONFIG_PATH = os.path.join(BASE_DIR, "config", "pipeline.yaml")

# Only read by the old dashboard (src/dashboard/app.py), which is due to be replaced.
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"
