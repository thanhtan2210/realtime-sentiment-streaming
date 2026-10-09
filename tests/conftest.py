import pandas as pd
import pytest

TEST_START = "2015-02-23T00:00:00-08:00"

WORDS = {
    "negative": ["delayed", "cancelled", "rude", "lost", "terrible", "worst"],
    "neutral": ["question", "schedule", "gate", "booking", "flying", "tomorrow"],
    "positive": ["thanks", "great", "awesome", "love", "friendly", "smooth"],
}
AIRLINES = ["United", "Delta", "Southwest"]


def _row(i, day, hour, label, text=None, minute=0, second=0):
    words = WORDS.get(label, WORDS["neutral"])
    return {
        "tweet_id": str(570000000000000000 + i),
        "airline_sentiment": label,
        "airline": AIRLINES[i % len(AIRLINES)],
        "text": text or f"@airline {words[i % 6]} {words[(i // 6) % 6]} flight {100 + i}",
        "tweet_created": f"2015-02-{day:02d} {hour:02d}:{minute:02d}:{second:02d} -0800",
    }


@pytest.fixture
def raw():
    """A small stand-in for data/raw/Tweets.csv: 8 days, 15 tweets a day."""
    rows = []
    labels = list(WORDS)
    i = 0
    for day in range(17, 25):
        for k in range(15):
            # negative is the majority (7 of 15), like the real data
            label = "negative" if k % 2 == 0 else labels[1 + (k // 2) % 2]
            rows.append(_row(i, day, 8 + k % 12, label, minute=k))
            i += 1

    # Same text before and after the cut-off, differing only in whitespace.
    rows.append(_row(900, 18, 9, "negative", text="my bag is   lost again", minute=30))
    rows.append(_row(901, 23, 9, "negative", text=" my bag is lost again ", minute=30))
    # One second either side of the cut-off.
    rows.append(_row(902, 22, 23, "neutral", text="last tweet before the cut", minute=59, second=59))
    rows.append(_row(903, 23, 0, "neutral", text="first tweet after the cut"))
    # A label outside the three classes.
    rows.append(_row(904, 19, 10, "mixed", text="not sure how I feel"))
    return pd.DataFrame(rows)


@pytest.fixture
def cfg():
    return {
        "split": {"test_start": TEST_START},
        "cv": {
            "n_splits": 3,
            "grid": {
                "tfidf__ngram_range": [[1, 1], [1, 2]],
                "tfidf__min_df": [1],
                "clf__C": [1.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "bootstrap": {"n_resamples": 50},
        "seed": 42,
    }
