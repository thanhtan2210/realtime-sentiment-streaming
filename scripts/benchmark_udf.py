"""Micro-benchmark: row-at-a-time `udf` vs vectorised `pandas_udf` for model inference.

Usage, inside the Spark image so it runs on the processor's own runtime:
    docker compose run --rm --no-deps \\
        -v ./scripts:/app/scripts:ro -v ./data/raw:/app/data/raw:ro -v ./reports:/app/reports \\
        processor python -m scripts.benchmark_udf

A Spark batch job scores the test tweets (repeated --rounds times) with each
variant and discards the output (`noop` sink), so only reading the cached rows
and running the model is timed. Each variant gets one untimed warm-up run, then
--repeats timed runs, alternating; the median is reported. Writes the
`udf_vs_pandas_udf` section of reports/benchmark.json.
"""
import argparse
import platform
import statistics
import sys
import time

import pandas as pd
import pyarrow
import pyspark
import sklearn
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from scripts.benchmark_report import REPORT_PATH, commit_sha, machine_info, update_report, utc_now
from src.ml.data import load_config, load_raw
from src.stream.producer import select_test_tweets
from src.stream.settings import stream_settings
from src.stream.transform import get_model, predict_sentiment


@udf(StringType())
def predict_sentiment_row(text):
    """Plain Python UDF: one model call per row."""
    return str(get_model().predict([text or ""])[0])


VARIANTS = {"udf": predict_sentiment_row, "pandas_udf": predict_sentiment}


def timed_run(tweets, variant) -> float:
    started = time.perf_counter()
    scored = tweets.select(variant(F.col("text")).alias("sentiment"))
    scored.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5, help="copies of the test set to score")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per variant")
    parser.add_argument("--limit", type=int, default=None, help="use only the first K test tweets")
    parser.add_argument("--output", default=REPORT_PATH)
    args = parser.parse_args()

    machine = machine_info()
    cfg = load_config()
    spark_cfg = stream_settings(cfg)["spark"]
    test = select_test_tweets(load_raw(), cfg, args.limit)
    texts = pd.DataFrame({"text": list(test["text"]) * args.rounds})

    spark = (
        SparkSession.builder.appName("udf-benchmark")
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    partitions = spark.sparkContext.defaultParallelism
    tweets = spark.createDataFrame(texts).repartition(partitions).cache()
    rows = tweets.count()

    # Both variants must agree before their speed is compared.
    sample = tweets.limit(500).select(
        predict_sentiment_row("text").alias("row"), predict_sentiment("text").alias("vectorised")
    )
    if any(row["row"] != row["vectorised"] for row in sample.collect()):
        print("udf and pandas_udf disagree; not measuring", file=sys.stderr)
        return 1

    warmup = {name: round(timed_run(tweets, variant), 3) for name, variant in VARIANTS.items()}
    runs = {name: [] for name in VARIANTS}
    for _ in range(args.repeats):
        for name, variant in VARIANTS.items():
            runs[name].append(round(timed_run(tweets, variant), 3))

    medians = {name: statistics.median(values) for name, values in runs.items()}
    result = {
        "measured_at_utc": utc_now(),
        "commit": commit_sha(),
        "command": "python -m scripts.benchmark_udf " + " ".join(sys.argv[1:]),
        "machine": machine,
        "what": "Spark batch job scoring every row with the sentiment model; output discarded (noop sink)",
        "rows": rows,
        "test_tweets": len(test),
        "rounds": args.rounds,
        "partitions": partitions,
        "spark_master": spark_cfg["master"],
        "spark_driver_memory": spark_cfg["driver_memory"],
        "arrow_max_records_per_batch": int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")),
        "timed_runs_per_variant": args.repeats,
        "warmup_seconds": warmup,
        "run_seconds": runs,
        "median_seconds": medians,
        "median_rows_per_second": {name: round(rows / median) for name, median in medians.items()},
        "pandas_udf_speedup_over_udf": round(medians["udf"] / medians["pandas_udf"], 2),
        "versions": {
            "python": platform.python_version(),
            "java": spark._jvm.System.getProperty("java.version"),
            "pyspark": pyspark.__version__,
            "pyarrow": pyarrow.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    spark.stop()
    update_report("udf_vs_pandas_udf", result, args.output)
    print(f"udf {medians['udf']}s, pandas_udf {medians['pandas_udf']}s (median of {args.repeats}, {rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
