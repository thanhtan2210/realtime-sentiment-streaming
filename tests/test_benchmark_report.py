"""The benchmark's arithmetic, on made-up data (no Docker, Kafka, Spark or MongoDB)."""
import json

import numpy as np
import pytest

from scripts.benchmark_report import (
    duration_stats,
    latency_seconds,
    machine_info,
    percentiles,
    throughput,
    update_report,
)


def test_percentiles_of_a_known_distribution():
    result = percentiles(range(1, 101))  # 1..100
    assert result == {"p50": 50.5, "p95": 95.05, "p99": 99.01, "min": 1.0, "max": 100.0}
    assert percentiles([2.5]) == {"p50": 2.5, "p95": 2.5, "p99": 2.5, "min": 2.5, "max": 2.5}
    with pytest.raises(ValueError):
        percentiles([])


def test_throughput_is_records_over_first_send_to_last_write():
    produced = np.array([1_000, 1_500, 2_000, 2_500])
    written = np.array([3_000, 3_000, 5_000, 5_000])
    # 4 records between the first send (t=1.0s) and the last write (t=5.0s)
    assert throughput(produced, written) == {
        "records": 4,
        "first_send_to_last_write_seconds": 4.0,
        "records_per_second": 1.0,
    }


def test_throughput_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        throughput([], [])
    with pytest.raises(ValueError):
        throughput([1_000, 2_000], [3_000])
    with pytest.raises(ValueError):
        throughput([5_000], [5_000])


def test_latency_is_per_record_and_in_seconds():
    produced = np.arange(100) * 10  # one record every 10 ms
    written = produced + np.arange(1, 101) * 100  # delays of 0.1 s .. 10.0 s
    assert latency_seconds(produced, written) == {
        "p50": 5.05,
        "p95": 9.505,
        "p99": 9.901,
        "min": 0.1,
        "max": 10.0,
    }


def test_duration_stats_give_median_p95_and_total_per_step():
    batches = [
        {"durationMs": {"addBatch": add, "triggerExecution": add + 100, "walCommit": 5}}
        for add in (100, 200, 300, 400, 1000)
    ]
    stats = duration_stats(batches)
    assert set(stats) == {"addBatch", "triggerExecution"}
    assert stats["addBatch"] == {"median_ms": 300.0, "p95_ms": 880.0, "total_ms": 2000}
    assert stats["triggerExecution"] == {"median_ms": 400.0, "p95_ms": 980.0, "total_ms": 2500}


def test_each_script_replaces_only_its_own_section(tmp_path):
    path = str(tmp_path / "reports" / "benchmark.json")
    update_report("streaming", {"records": 1}, path)
    update_report("udf_vs_pandas_udf", {"rows": 2}, path)
    update_report("streaming", {"records": 3}, path)
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["streaming"] == {"records": 3}
    assert report["udf_vs_pandas_udf"] == {"rows": 2}
    assert "note" in report


def test_machine_info_describes_this_machine():
    info = machine_info()
    assert info["nproc"] >= 1
    assert info["ram_total_gb"] > 0
    assert info["cpu_model"]
    assert isinstance(info["github_runner"], bool)
