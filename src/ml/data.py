"""Load the raw tweets, drop duplicate texts, and split train/test by time."""
import pandas as pd
import yaml

from src.common import config

LABELS = ["negative", "neutral", "positive"]
COLUMNS = ["tweet_id", "created_at", "airline", "text", "label"]


def load_config(path: str = config.PIPELINE_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(path: str = config.CSV_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"tweet_id": str}, encoding="utf-8")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the modelling columns, order by time and drop duplicates.

    Duplicate texts (compared after collapsing whitespace) go first, then any
    remaining duplicate tweet_id; the earliest copy is kept each time.
    This runs on the full dataset, before any train/test split.
    """
    df = pd.DataFrame(
        {
            "tweet_id": raw["tweet_id"].astype(str),
            "created_at": pd.to_datetime(
                raw["tweet_created"], format="%Y-%m-%d %H:%M:%S %z", utc=True
            ),
            "airline": raw["airline"],
            "text": raw["text"].fillna("").astype(str).str.split().str.join(" "),
            "label": raw["airline_sentiment"],
        }
    )
    df = df[df["label"].isin(LABELS) & (df["text"] != "")]
    rows_with_valid_label = len(df)

    df = df.sort_values(["created_at", "tweet_id"], kind="stable")
    labels_per_text = df.groupby("text")["label"].nunique()
    by_text = df.drop_duplicates(subset="text", keep="first")
    # tweet_id becomes the document key downstream, so it must be unique too.
    deduped = by_text.drop_duplicates(subset="tweet_id", keep="first").reset_index(drop=True)

    stats = {
        "raw_rows": int(len(raw)),
        "rows_with_valid_label_and_text": int(rows_with_valid_label),
        "duplicate_text_rows_removed": int(rows_with_valid_label - len(by_text)),
        "duplicated_texts_with_conflicting_labels": int((labels_per_text > 1).sum()),
        "duplicate_tweet_id_rows_removed": int(len(by_text) - len(deduped)),
        "rows": int(len(deduped)),
    }
    return deduped[COLUMNS], stats


def time_split(df: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train is everything before `test_start`; test is everything from it on."""
    cutoff = pd.Timestamp(test_start)
    if cutoff.tzinfo is None:
        raise ValueError("split.test_start must include a UTC offset")
    is_test = df["created_at"] >= cutoff
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def prepare(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, stats = clean(raw)
    train, test = time_split(df, cfg["split"]["test_start"])
    return train, test, stats
