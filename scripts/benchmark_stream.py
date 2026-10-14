"""Benchmark the running stack; writes the `streaming` section of reports/benchmark.json.

Usage (from the repo root, after `docker compose up -d` on a fresh stack):
    python -m scripts.benchmark_stream

After a small untimed warm-up (the first batch pays for JVM and Python worker
start-up and for loading the model), the whole test set is replayed --rounds
times at max rate, one run-id per round, and the script waits until MongoDB
holds every prediction. Round k shifts the tweets' event time by 3*k days
(the warm-up by -3 days), so no round is late for the watermark and every
tweet goes through the window aggregation. From the stored documents:

  throughput  records / (last written_at - first produced_at_ms)
  latency     p50 / p95 / p99 of written_at - produced_at_ms, and of the same
              plus the MongoDB write time of the record's micro-batch
              (i.e. until MongoDB acknowledged the batch)

`_id` is the tweet id, not an ObjectId, so MongoDB keeps no insert time of its
own; the acknowledgement time comes from the processor's `WRITE` log lines.
The producer runs in a container, so produced_at_ms and written_at come from
the same clock. Batch step timings come from the `PROGRESS` log lines
(Spark's StreamingQueryProgress.durationMs).

The run ends with two checks, written to the report: the window totals grew by
exactly the number of timed tweets, and the watermark dropped no row. The
script exits with code 1 if either fails.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from pymongo import MongoClient

from scripts.benchmark_report import (
    REPORT_PATH,
    commit_sha,
    duration_stats,
    latency_seconds,
    machine_info,
    throughput,
    update_report,
    utc_now,
)
from src.ml.data import load_config, load_raw
from src.stream.producer import select_test_tweets
from src.stream.settings import stream_settings

WARMUP_RECORDS = 200
WARMUP_SHIFT_DAYS = -3  # its own event-time range, before round 0
ROUND_SHIFT_DAYS = 3  # the test set spans about 36 hours


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        sys.exit(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-2000:]}")
    return result.stdout + result.stderr


def docker_info() -> dict:
    docker = json.loads(run(["docker", "info", "--format", "{{json .}}"]).splitlines()[0])
    compose = json.loads(run(["docker", "compose", "--profile", "replay", "config", "--format", "json"]))
    services = compose["services"]
    return {
        "docker_version": docker["ServerVersion"],
        "docker_memory_gb": round(docker["MemTotal"] / 2**30, 2),
        "docker_cpus": docker["NCPU"],
        "images": {name: service["image"] for name, service in services.items()},
        "container_memory_limits_mb": {
            name: int(service["mem_limit"]) // 2**20
            for name, service in services.items()
            if "mem_limit" in service
        },
    }


def processor_versions() -> dict:
    """Library versions inside the running processor container."""
    code = (
        "import json, pyspark, pyarrow, pandas, sklearn, platform, glob;"
        "print('VERSIONS ' + json.dumps({'python': platform.python_version(),"
        " 'pyspark': pyspark.__version__, 'pyarrow': pyarrow.__version__,"
        " 'pandas': pandas.__version__, 'scikit-learn': sklearn.__version__,"
        " 'spark_connector_jars': sorted(p.rsplit('/', 1)[-1] for p in glob.glob('/opt/spark/jars/*')"
        " if 'spark' in p.rsplit('/', 1)[-1])}))"
    )
    output = run(["docker", "compose", "exec", "-T", "processor", "python", "-c", code])
    line = next(line for line in output.splitlines() if line.startswith("VERSIONS "))
    return json.loads(line[len("VERSIONS "):])


def topic_partitions(topic: str) -> int:
    """Partition count of the topic, asked from inside the compose network."""
    code = (
        "import os; from kafka import KafkaConsumer;"
        "c = KafkaConsumer(bootstrap_servers=os.environ['KAFKA_BOOTSTRAP_SERVERS']);"
        f"print('PARTITIONS', len(c.partitions_for_topic('{topic}')))"
    )
    output = run(["docker", "compose", "run", "--rm", "--entrypoint", "python", "producer", "-c", code])
    line = next(line for line in output.splitlines() if line.startswith("PARTITIONS "))
    return int(line.split()[1])


def processor_log_events() -> tuple[list[dict], list[dict]]:
    """(PROGRESS events, WRITE events) parsed from the processor's log."""
    progress, writes = [], []
    for line in run(["docker", "compose", "logs", "--no-log-prefix", "processor"]).splitlines():
        for marker, target in (("PROGRESS ", progress), ("WRITE ", writes)):
            if marker + "{" in line:
                target.append(json.loads(line.split(marker, 1)[1]))
    return progress, writes


def to_ms(documents: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(produced_at_ms, written_at in epoch ms) of prediction documents."""
    produced = np.array([d["produced_at_ms"] for d in documents], dtype=np.int64)
    written = np.array(
        [d["written_at"].replace(tzinfo=timezone.utc).timestamp() * 1000 for d in documents]
    )
    return produced, written.round().astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5, help="replays of the test set")
    parser.add_argument("--limit", type=int, default=None, help="send only the first K test tweets per round")
    parser.add_argument("--timeout", type=int, default=600, help="seconds to wait for MongoDB")
    parser.add_argument("--output", default=REPORT_PATH)
    args = parser.parse_args()

    machine = machine_info()
    cfg = load_config()
    settings = stream_settings(cfg)
    mongodb = settings["mongodb"]
    per_round = len(select_test_tweets(load_raw(), cfg, args.limit))
    expected = per_round * args.rounds

    client = MongoClient(mongodb["uri"], serverSelectionTimeoutMS=10_000)
    mongodb_version = client.server_info()["version"]
    predictions = client[mongodb["database"]][mongodb["predictions_collection"]]
    windows = client[mongodb["database"]][mongodb["windows_collection"]]

    # Windows of the timed rounds: everything from the first test tweet's hour
    # on. The warm-up, shifted back, falls before it.
    first_event = select_test_tweets(load_raw(), cfg, args.limit)["created_at"].min()
    timed_windows = {"window_start": {"$gte": first_event.floor("h").to_pydatetime()}}

    def window_total() -> int:
        grouped = windows.aggregate(
            [{"$match": timed_windows}, {"$group": {"_id": None, "total": {"$sum": "$total"}}}]
        )
        return next((int(row["total"]) for row in grouped), 0)

    window_total_before = window_total()
    partitions = topic_partitions(settings["kafka"]["topic"])

    def wait_for(query: dict, count: int) -> list[dict]:
        deadline = time.monotonic() + args.timeout
        while (stored := predictions.count_documents(query)) < count:
            if time.monotonic() > deadline:
                sys.exit(f"Timed out: {stored} of {count} predictions in MongoDB after {args.timeout}s")
            time.sleep(1)
        return list(predictions.find(query, {"produced_at_ms": 1, "written_at": 1}))

    bench_id = datetime.now(timezone.utc).strftime("b%Y%m%dT%H%M%S")
    limit = ["--limit", str(args.limit)] if args.limit else []

    warmup_command = ["docker", "compose", "run", "--rm", "producer", "--max-rate",
                      "--limit", str(WARMUP_RECORDS), "--run-id", f"{bench_id}warmup",
                      "--time-shift-days", str(WARMUP_SHIFT_DAYS)]  # fmt: skip
    print(f"Warm-up: {WARMUP_RECORDS} tweets ...")
    run(warmup_command)
    warm_produced, warm_written = to_ms(wait_for({"_id": {"$regex": f"-{bench_id}warmup$"}}, WARMUP_RECORDS))
    time.sleep(10)  # let the warm-up batch finish its write before the timed rounds

    # One container, N producer runs back to back: no container start-up between rounds.
    loop = " && ".join(
        " ".join(["python", "-m", "src.stream.producer", "--max-rate", *limit,
                  "--run-id", f"{bench_id}r{k}", "--time-shift-days", str(ROUND_SHIFT_DAYS * k)])
        for k in range(args.rounds)
    )  # fmt: skip
    command = ["docker", "compose", "run", "--rm", "--entrypoint", "sh", "producer", "-c", loop]
    print(f"Sending {args.rounds} x {per_round} tweets at max rate ...")
    run(command)
    produced, written = to_ms(wait_for({"_id": {"$regex": f"-{bench_id}r[0-9]+$"}}, expected))
    # The window query is a separate stream; give it up to a minute to catch up.
    deadline = time.monotonic() + 60
    while window_total() - window_total_before < expected and time.monotonic() < deadline:
        time.sleep(1)
    time.sleep(10)  # let the last progress events reach the processor's log
    window_total_added = window_total() - window_total_before

    # A micro-batch stamps all its documents with one written_at, which the
    # processor also logs: that links each document to its batch.
    progress, writes = processor_log_events()
    write_by_stamp = {
        round(datetime.fromisoformat(w["writtenAt"]).timestamp() * 1000): w
        for w in writes
        if w["sink"] == "predictions"
    }
    stamps = sorted(set(written.tolist()))
    missing = [stamp for stamp in stamps if stamp not in write_by_stamp]
    if missing:
        sys.exit(f"{len(missing)} written_at values have no WRITE line in the processor log")
    batch_writes = [write_by_stamp[stamp] for stamp in stamps]
    batch_ids = {w["batchId"] for w in batch_writes}
    batches = [p for p in progress if p["name"] == "predictions" and p["batchId"] in batch_ids]
    first_batch_started = min(b["timestamp"] for b in batches)
    window_batches = [
        p for p in progress if p["name"] == "airline_windows" and p["timestamp"] >= first_batch_started
    ]
    acknowledged = written + np.array([write_by_stamp[stamp]["mongoWriteMs"] for stamp in written.tolist()])
    dropped_by_watermark = sum(b["rowsDroppedByWatermark"] for b in window_batches)
    checks = {
        "timed_tweets_sent": expected,
        "window_total_added": window_total_added,
        "window_total_equals_timed_tweets": window_total_added == expected,
        "rows_dropped_by_watermark": dropped_by_watermark,
        "no_rows_dropped_by_watermark": dropped_by_watermark == 0,
    }
    checks["passed"] = checks["window_total_equals_timed_tweets"] and checks["no_rows_dropped_by_watermark"]

    result = {
        "measured_at_utc": utc_now(),
        "commit": commit_sha(),
        "command": "python -m scripts.benchmark_stream " + " ".join(sys.argv[1:]),
        "machine": machine,
        "docker": docker_info(),
        "versions": {**processor_versions(), "mongodb_server": mongodb_version},
        "settings": {
            "trigger": settings["trigger"],
            "window": settings["window"],
            "watermark": settings["watermark"],
            "topic_partitions": partitions,
            "shuffle_partitions": settings["spark"]["shuffle_partitions"],
        },
        "warmup": {
            "command": " ".join(warmup_command),
            "records": WARMUP_RECORDS,
            "included_in_the_results": False,
            "time_shift_days": WARMUP_SHIFT_DAYS,
            "first_send_to_last_write_seconds": round(
                (int(warm_written.max()) - int(warm_produced.min())) / 1000, 3
            ),
        },
        "load": {
            "command": " ".join(command[:-1]) + f' "{loop}"',
            "rounds": args.rounds,
            "records_per_round": per_round,
            "time_shift_days_per_round": [ROUND_SHIFT_DAYS * k for k in range(args.rounds)],
            "produce_seconds": round((int(produced.max()) - int(produced.min())) / 1000, 3),
        },
        "checks": checks,
        "throughput": throughput(produced, written),
        "latency_seconds": {
            "to_written_at": latency_seconds(produced, written),
            "to_mongodb_acknowledgement": latency_seconds(produced, acknowledged),
        },
        "definitions": {
            "throughput": "records / (last written_at - first produced_at_ms)",
            "produced_at_ms": "set by the producer just before it hands the message to the Kafka client",
            "written_at": "set by the Spark driver after the model ran on the micro-batch, "
            "immediately before its MongoDB write starts",
            "to_mongodb_acknowledgement": "written_at plus the time the batch's MongoDB write took "
            "(mongoWriteMs in the processor log)",
        },
        "predictions_query": {
            "batches": len(batches),
            "rows_per_batch": [b["numInputRows"] for b in batches],
            "duration_ms": duration_stats(batches),
            "read_and_predict_ms_total": sum(w["readAndPredictMs"] for w in batch_writes),
            "mongo_write_ms_total": sum(w["mongoWriteMs"] for w in batch_writes),
        },
        "airline_windows_query": {
            "batches": len(window_batches),
            "input_rows": sum(b["numInputRows"] for b in window_batches),
            "rows_dropped_by_watermark": dropped_by_watermark,
            "duplicates_dropped": sum(b["duplicatesDropped"] for b in window_batches),
            "duration_ms": duration_stats(window_batches),
        },
    }
    update_report("streaming", result, args.output)
    print(json.dumps({"throughput": result["throughput"], "latency_seconds": result["latency_seconds"]}))
    print(json.dumps({"checks": checks}))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
