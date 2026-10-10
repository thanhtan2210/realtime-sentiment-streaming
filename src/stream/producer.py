"""Replay the test-set tweets into Kafka.

Usage (from the repo root, with the stack running):
    python -m src.stream.producer --speedup 600      # 600x faster than real time
    python -m src.stream.producer --max-rate --limit 500

Only the test split is replayed: the model never saw these tweets. Labels are
not sent.
"""
import argparse
import sys
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from src.common import logger as common_logger
from src.ml.data import load_config, load_raw, prepare
from src.stream.settings import stream_settings

logger = common_logger.get_logger(__name__)


class TweetMessage(BaseModel):
    """The JSON value written to Kafka. Deliberately has no label field."""

    model_config = ConfigDict(extra="forbid")

    tweet_id: str = Field(min_length=1)
    airline: str = Field(min_length=1)
    text: str = Field(min_length=1)
    created_at: AwareDatetime
    produced_at_ms: int = Field(gt=0)


def select_test_tweets(raw: pd.DataFrame, cfg: dict, limit: int | None = None) -> pd.DataFrame:
    """The test split from src.ml.data (same cut-off as training), ordered by time."""
    _, test, _ = prepare(raw, cfg)
    test = test.sort_values("created_at", kind="stable").reset_index(drop=True)
    return test if limit is None else test.head(limit)


def build_message(row, run_id: str | None, produced_at_ms: int) -> TweetMessage:
    tweet_id = f"{row.tweet_id}-{run_id}" if run_id else row.tweet_id
    return TweetMessage(
        tweet_id=tweet_id,
        airline=row.airline,
        text=row.text,
        created_at=row.created_at,
        produced_at_ms=produced_at_ms,
    )


def serialize(message: TweetMessage) -> tuple[bytes, bytes]:
    """Kafka (key, value): the tweet id, and the message as JSON.

    Keying by tweet id spreads tweets evenly over the partitions and sends any
    re-sent copy of a tweet to the same partition as the original.
    """
    return message.tweet_id.encode("utf-8"), message.model_dump_json().encode("utf-8")


def send_offsets(created_at: pd.Series, speedup: float) -> list[float]:
    """Seconds after the start at which each tweet is due: event-time gaps divided by `speedup`."""
    return ((created_at - created_at.iloc[0]).dt.total_seconds() / speedup).tolist()


def replay(producer, topic: str, tweets: pd.DataFrame, speedup: float | None, run_id: str | None):
    """Send every tweet; `speedup=None` means as fast as possible. Returns (sent, failed)."""
    counts = {"sent": 0, "failed": 0}

    def on_success(_metadata):
        counts["sent"] += 1

    def on_error(exc):
        counts["failed"] += 1
        logger.error(f"Send failed: {exc}")

    offsets = send_offsets(tweets["created_at"], speedup) if speedup else None
    start = time.monotonic()
    for i, row in enumerate(tweets.itertuples(index=False)):
        if offsets:
            wait = start + offsets[i] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        try:
            key, value = serialize(build_message(row, run_id, int(time.time() * 1000)))
        except ValidationError as exc:
            counts["failed"] += 1
            logger.warning(f"Invalid tweet {row.tweet_id} skipped: {exc.errors()}")
            continue
        producer.send(topic, key=key, value=value).add_callback(on_success).add_errback(on_error)

    producer.flush()
    return counts["sent"], counts["failed"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay the test-set tweets into Kafka.")
    pace = parser.add_mutually_exclusive_group()
    pace.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="replay with the original gaps between tweets divided by N (default 1)",
    )
    pace.add_argument(
        "--max-rate",
        action="store_true",
        help="ignore the gaps and send as fast as possible (benchmarks)",
    )
    parser.add_argument("--limit", type=int, default=None, help="send only the first K tweets")
    parser.add_argument(
        "--run-id",
        default=None,
        help="suffix added to tweet_id so repeated runs do not share keys",
    )
    args = parser.parse_args(argv)
    if args.speedup <= 0:
        parser.error("--speedup must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    kafka = stream_settings(cfg)["kafka"]
    tweets = select_test_tweets(load_raw(), cfg, args.limit)
    speedup = None if args.max_rate else args.speedup

    try:
        producer = KafkaProducer(bootstrap_servers=kafka["bootstrap_servers"], acks="all")
    except KafkaError as exc:
        logger.error(f"Cannot connect to Kafka at {kafka['bootstrap_servers']}: {exc!r}")
        return 1

    pace = "max rate" if speedup is None else f"speedup {speedup:g}"
    logger.info(f"Replaying {len(tweets)} test tweets to '{kafka['topic']}' ({pace})")
    started = time.monotonic()
    sent, failed = replay(producer, kafka["topic"], tweets, speedup, args.run_id)
    producer.close()
    logger.info(f"Done in {time.monotonic() - started:.1f}s: sent={sent} failed={failed}")
    return 0 if failed == 0 and sent == len(tweets) else 1


if __name__ == "__main__":
    sys.exit(main())
