"""Shared by the benchmark scripts: the statistics, the machine description and the report file.

reports/benchmark.json has one section per script (`streaming`, `udf_vs_pandas_udf`);
each script replaces only its own section. The numbers are a one-off measurement of
the machine that ran them, so the file is allowed to differ between machines.
"""
import ctypes
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "reports", "benchmark.json")


def percentiles(values) -> dict:
    """p50 / p95 / p99 (plus min and max) of a non-empty sequence, rounded to 3 decimals."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("no values")
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {
        "p50": round(float(p50), 3),
        "p95": round(float(p95), 3),
        "p99": round(float(p99), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def throughput(produced_ms, written_ms) -> dict:
    """Records per second from the first send to the last write (both in epoch milliseconds)."""
    produced_ms, written_ms = np.asarray(produced_ms), np.asarray(written_ms)
    if produced_ms.size == 0 or produced_ms.size != written_ms.size:
        raise ValueError("need one produced and one written timestamp per record")
    span_s = (int(written_ms.max()) - int(produced_ms.min())) / 1000
    if span_s <= 0:
        raise ValueError("the last write is not after the first send")
    return {
        "records": int(produced_ms.size),
        "first_send_to_last_write_seconds": round(span_s, 3),
        "records_per_second": round(produced_ms.size / span_s, 1),
    }


def latency_seconds(produced_ms, written_ms) -> dict:
    """Percentiles of (written - produced) per record, in seconds."""
    return percentiles((np.asarray(written_ms) - np.asarray(produced_ms)) / 1000)


def duration_stats(batches: list[dict], steps=("addBatch", "triggerExecution")) -> dict:
    """Median, p95 and total milliseconds of the given StreamingQueryProgress.durationMs steps."""
    stats = {}
    for step in steps:
        values = [batch["durationMs"].get(step, 0) for batch in batches]
        stats[step] = {
            "median_ms": round(float(np.percentile(values, 50)), 1),
            "p95_ms": round(float(np.percentile(values, 95)), 1),
            "total_ms": int(sum(values)),
        }
    return stats


def _memory_gb() -> tuple[float, float]:
    """(total, available) physical memory in GiB."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
    with open("/proc/meminfo", encoding="utf-8") as f:
        info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return info["MemTotal"] / 2**20, info["MemAvailable"] / 2**20


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        return next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "")


def _os_name() -> str:
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        # Windows 11 still reports itself as version 10.0; the build number tells them apart.
        return f"Windows {'11' if build >= 22000 else '10'} (build {build})"
    return platform.platform()


def machine_info() -> dict:
    total_gb, available_gb = _memory_gb()
    return {
        "nproc": os.cpu_count(),
        "ram_total_gb": round(total_gb, 1),
        "ram_available_at_start_gb": round(available_gb, 1),
        "cpu_model": _cpu_model(),
        "os": _os_name(),
        "inside_container": os.path.exists("/.dockerenv"),
        "github_runner": os.environ.get("GITHUB_ACTIONS") == "true",
        "github_runner_image": os.environ.get("ImageOS"),
    }


def commit_sha() -> str:
    """The commit being measured: BENCHMARK_COMMIT or GITHUB_SHA if set, else `git rev-parse HEAD`."""
    for variable in ("BENCHMARK_COMMIT", "GITHUB_SHA"):
        if os.environ.get(variable):
            return os.environ[variable]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def update_report(section: str, data: dict, path: str = REPORT_PATH) -> None:
    """Replace one section of the report file, keeping the others."""
    report = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    report["note"] = (
        "One-off measurements of the machine described in each section; they are not "
        "reproducible to the digit. See scripts/benchmark_stream.py and scripts/benchmark_udf.py."
    )
    report[section] = data
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote section '{section}' to {path}")
