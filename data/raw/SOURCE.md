# Data source

| | |
|---|---|
| Dataset | Twitter US Airline Sentiment |
| Original publisher | Crowdflower / Figure Eight, "Data for Everyone" library |
| Dataset page | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| Downloaded from | https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/resolve/dec17e9391b767791e3808a655654467605a9d49/Tweets.csv |
| Downloaded on | 2026-10-09 (UTC) |
| License | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| File | `Tweets.csv` |
| Size | 3421431 bytes |
| Rows | 14640 |
| sha256 | `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e` |

The download URL is a public Hugging Face mirror of the Kaggle dataset
(`osanseviero/twitter-airline-sentiment`, converted from Kaggle), pinned to one revision.
The license is the one stated on the Kaggle dataset page and repeated on
the mirror's dataset card. It is non-commercial and share-alike.

## What the license covers

`Tweets.csv`, and anything produced from it that contains tweet text, is under
CC BY-NC-SA 4.0: exports or snapshots of the stream (predictions stored with
their `text`), samples, and screenshots that show tweets.
`models/sentiment_pipeline.joblib` stores a vocabulary of words and word pairs
taken from the tweets, so treat it the same way.

The code in this repository is under the MIT license (see `LICENSE`).

Written by `scripts/fetch_data.py`. Do not edit by hand.
