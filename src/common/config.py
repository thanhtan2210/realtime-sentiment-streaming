import os

# Paths. Kafka, MongoDB, Spark and dashboard settings live in config/pipeline.yaml.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
CSV_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Tweets.csv")
PIPELINE_CONFIG_PATH = os.path.join(BASE_DIR, "config", "pipeline.yaml")
