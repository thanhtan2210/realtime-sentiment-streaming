import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Setup path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import config, logger as common_logger

logger = common_logger.get_logger(__name__)

DEFAULT_TEXT_CANDIDATES = ["text", "review", "comment", "content", "sentence", "message"]
DEFAULT_LABEL_CANDIDATES = ["label", "sentiment", "target", "class"]

def detect_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None) -> tuple[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    tc = text_col or next((cols_lower.get(c) for c in DEFAULT_TEXT_CANDIDATES if c in cols_lower), None)
    lc = label_col or next((cols_lower.get(c) for c in DEFAULT_LABEL_CANDIDATES if c in cols_lower), None)
    if not tc or not lc:
        raise ValueError("Could not detect columns.")
    return tc, lc

def main():
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--input", type=str, default=config.CSV_DATA_PATH)
    parser.add_argument("--model-out", type=str, default=config.MODEL_PATH)
    args = parser.parse_args()

    df = pd.read_csv(args.input).dropna(how="any")
    text_col, label_col = detect_columns(df, None, None)

    X = df[text_col].astype(str)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)

    logger.info(f"Training completed. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
