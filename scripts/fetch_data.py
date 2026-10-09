"""Download the Twitter US Airline Sentiment dataset into data/raw/.

Usage (from the repo root):
    python scripts/fetch_data.py

The file is fetched from a public Hugging Face mirror of the Kaggle dataset
(no login or API key), pinned to one git revision and checked against a
pinned sha256. data/raw/SOURCE.md records where and when it was downloaded.
If the file is already present with the expected hash, nothing is rewritten.
"""
import csv
import hashlib
import io
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HF_REPO = "osanseviero/twitter-airline-sentiment"
HF_REVISION = "dec17e9391b767791e3808a655654467605a9d49"
FILENAME = "Tweets.csv"
URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{FILENAME}"
EXPECTED_SHA256 = "ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e"

KAGGLE_URL = "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment"
LICENSE_NAME = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CSV_PATH = RAW_DIR / FILENAME
SOURCE_PATH = RAW_DIR / "SOURCE.md"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_rows(data: bytes) -> int:
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    return sum(1 for _ in reader) - 1  # minus header


def write_source(data: bytes, downloaded_on: str) -> None:
    SOURCE_PATH.write_text(
        "\n".join(
            [
                "# Data source",
                "",
                "| | |",
                "|---|---|",
                "| Dataset | Twitter US Airline Sentiment |",
                "| Original publisher | Crowdflower / Figure Eight, \"Data for Everyone\" library |",
                f"| Dataset page | {KAGGLE_URL} |",
                f"| Downloaded from | {URL} |",
                f"| Downloaded on | {downloaded_on} (UTC) |",
                f"| License | [{LICENSE_NAME}]({LICENSE_URL}) |",
                f"| File | `{FILENAME}` |",
                f"| Size | {len(data)} bytes |",
                f"| Rows | {count_rows(data)} |",
                f"| sha256 | `{sha256_of(data)}` |",
                "",
                "The download URL is a public Hugging Face mirror of the Kaggle dataset",
                f"(`{HF_REPO}`, converted from Kaggle), pinned to one revision.",
                "The license is the one stated on the Kaggle dataset page and repeated on",
                "the mirror's dataset card. It is non-commercial and share-alike.",
                "",
                "Written by `scripts/fetch_data.py`. Do not edit by hand.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        data = CSV_PATH.read_bytes()
        if sha256_of(data) == EXPECTED_SHA256:
            print(f"{CSV_PATH} already present, sha256 OK: {EXPECTED_SHA256}")
            if not SOURCE_PATH.exists():
                write_source(data, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            return 0
        print(f"{CSV_PATH} exists but its sha256 does not match; downloading again.")

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()

    actual = sha256_of(data)
    if actual != EXPECTED_SHA256:
        print(f"sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}", file=sys.stderr)
        return 1

    CSV_PATH.write_bytes(data)
    write_source(data, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    print(f"Saved {CSV_PATH} ({len(data)} bytes, {count_rows(data)} rows)")
    print(f"sha256: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
