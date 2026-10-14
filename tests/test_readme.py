"""Keeps README.md in line with the files its numbers come from.

Every figure in the README must be found in one of NUMBER_SOURCES. To document a
new report (for example reports/benchmark.json), add its path to
OPTIONAL_SOURCES and add tests for its section; nothing here needs rewriting.
"""
import json
import os
import re

import pandas as pd
import pytest
import yaml

from src.common import config
from src.ml.data import load_config, load_raw, prepare

ROOT = config.BASE_DIR


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def load_json(*parts: str) -> dict:
    return json.loads(read(*parts))


README = read("README.md")
METRICS = load_json("models", "metrics.json")
SNAPSHOT = load_json("reports", "stream_snapshot", "meta.json")
CONSISTENCY = load_json("reports", "stream_consistency.json")
NUMBER_SOURCES = [METRICS, SNAPSHOT, CONSISTENCY]

# Reports that do not exist yet. Once the file is committed its numbers may appear in the README.
OPTIONAL_SOURCES = [("reports", "benchmark.json")]
for _parts in OPTIONAL_SOURCES:
    if os.path.exists(os.path.join(ROOT, *_parts)):
        NUMBER_SOURCES.append(load_json(*_parts))
BENCHMARK_EXISTS = os.path.exists(os.path.join(ROOT, "reports", "benchmark.json"))

SECTIONS = [
    "Live demo",
    "Problem → Approach → Results",
    "Results",
    "How it works",
    "How to run",
    "Limitations",
    "Authors and license",
]
BENCHMARK_PLACEHOLDER = (
    "Throughput and latency are measured on a GitHub Actions runner by "
    "`.github/workflows/benchmark.yml`; results will be added to `reports/benchmark.json`."
)


def section(title: str, level: int = 2) -> str:
    """The text under a heading, up to the next heading of the same or a higher level."""
    lines = README.splitlines()
    start = lines.index("#" * level + " " + title) + 1
    body, fenced = [], False
    for line in lines[start:]:
        if line.startswith("```"):
            fenced = not fenced
        # a "# comment" inside a code block is not a heading
        if not fenced and re.match(rf"#{{1,{level}}} ", line):
            break
        body.append(line)
    return "\n".join(body).strip()


def table(text: str) -> list[list[str]]:
    """The body rows of the first Markdown table in `text`, as lists of cell strings."""
    rows = [line for line in text.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


def numbers_in(node) -> list:
    if isinstance(node, dict):
        return [n for value in node.values() for n in numbers_in(value)]
    if isinstance(node, list):
        return [n for value in node for n in numbers_in(value)]
    return [node] if isinstance(node, (int, float)) and not isinstance(node, bool) else []


def score(value: float) -> str:
    return f"{value:.4f}"


def score_with_interval(value: float, interval: list[float]) -> str:
    return f"{score(value)} [{score(interval[0])}, {score(interval[1])}]"


# --- structure ------------------------------------------------------------------------------


def test_sections_follow_the_agreed_order():
    assert re.findall(r"^## (.+)$", README, flags=re.MULTILINE) == SECTIONS
    lines = README.splitlines()
    assert lines[0].startswith("# ")
    assert lines[2] == "Right now, are each airline's customers talking positively or negatively?"


def test_the_only_todo_is_the_live_demo_link():
    assert README.count("TODO") == 1
    assert "TODO: link" in section("Live demo")


def test_every_image_shown_exists():
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
    assert images == ["docs/images/dashboard.png", "docs/images/replay.gif"]
    for image in images:
        assert os.path.isfile(os.path.join(ROOT, image)), image
    assert "docs/images/dashboard.png" in section("Live demo")
    assert "docs/images/replay.gif" in section("How it works")


def test_summary_has_five_to_eight_lines():
    lines = [line for line in section("Problem → Approach → Results").splitlines() if line.startswith("- ")]
    assert 5 <= len(lines) <= 8


def test_no_unproven_marketing_words():
    lowered = README.lower()
    for phrase in (
        "industrial-grade",
        "production-ready",
        "real-time at scale",
        "enterprise-grade",
        "high availability",
        "state-of-the-art",
    ):
        assert phrase not in lowered, phrase


def test_every_results_table_names_a_script_that_exists():
    results = section("Results")
    modules = re.findall(r"written by `python -m ([\w.]+)`", results)
    assert modules == ["src.ml.train", "src.ml.train", "scripts.export_snapshot", "scripts.check_stream_consistency"]
    for module in modules:
        assert os.path.isfile(os.path.join(ROOT, *module.split(".")) + ".py"), module
    for path in re.findall(r"Source: `([^`]+)`", results):
        assert os.path.isfile(os.path.join(ROOT, path)), path


# --- every number comes from a report ----------------------------------------------------------


def test_every_score_and_every_large_count_is_in_a_report():
    known = numbers_in(NUMBER_SOURCES)
    scores = {score(n) for n in known}
    counts = {f"{n:,}" for n in known if isinstance(n, int)}
    for found in re.findall(r"\b\d\.\d{4}\b", README):
        assert found in scores, f"{found} is not in any report"
    for found in re.findall(r"\b\d{1,3}(?:,\d{3})+\b", README):
        assert found in counts, f"{found} is not in any report"


def test_summary_numbers():
    summary = section("Problem → Approach → Results")
    model, baseline = METRICS["test"]["model"], METRICS["test"]["baseline"]
    low, high = model["macro_f1_ci95"]
    assert f"{METRICS['data']['raw_rows']:,} labelled tweets about six US airlines" in summary
    assert len(METRICS["test"]["by_airline"]) == 6
    assert f"macro-F1 is {score(model['macro_f1'])} (95% CI {score(low)} to {score(high)})" in summary
    assert f"against {score(baseline['macro_f1'])} for always answering \"{METRICS['test']['baseline_predicts']}\"" in summary
    records = SNAPSHOT["records"]
    assert (
        f"sent {SNAPSHOT['tweets_sent']:,} tweets and produced {records['predictions']:,} predictions "
        f"and {records['airline_windows']} airline-hour windows; the watermark dropped "
        f"{SNAPSHOT['rows_dropped_by_watermark']} rows"
    ) in summary
    assert (
        f"Stream predictions match the offline model on {CONSISTENCY['same_prediction_as_offline_model']:,} "
        f"of {CONSISTENCY['tweets']:,} tweets"
    ) in summary


def test_model_table_matches_metrics_json():
    text = section("Model on the test set", level=3)
    test, split = METRICS["test"], METRICS["split"]
    best = next(row for row in METRICS["cv"]["results"] if row["rank"] == 1)
    assert f"trained on {split['train']['rows']:,} tweets" in text
    assert f"the {split['test']['rows']:,} tweets created from {split['test_start'][:10]}" in text
    assert f"best cross-validated macro-F1: {score(best['mean_macro_f1'])}" in text
    assert f"{test['bootstrap']['n_resamples']:,} resamples" in text

    def row(name: str, result: dict) -> list[str]:
        return [
            name,
            score(result["accuracy"]),
            score_with_interval(result["macro_f1"], result["macro_f1_ci95"]),
            *(score(result["f1_per_class"][label]) for label in test["labels"]),
        ]

    difference = test["macro_f1_difference"]
    assert table(text) == [
        row(f"Baseline: always \"{test['baseline_predicts']}\"", test["baseline"]),
        row("TF-IDF + logistic regression", test["model"]),
        ["Difference (model − baseline)", "", score_with_interval(difference["value"], difference["ci95"]), "", "", ""],
    ]


def test_airline_table_matches_metrics_json():
    text = section("Macro-F1 by airline", level=3)
    airlines = METRICS["test"]["by_airline"]
    assert table(text) == [
        [a["airline"], f"{a['n']:,}", score_with_interval(a["macro_f1"], a["macro_f1_ci95"])] for a in airlines
    ]
    # the sentence under the table names the airlines whose largest class is neutral
    largest = {a["airline"]: max(a["label_counts"], key=a["label_counts"].get) for a in airlines}
    assert set(largest.values()) == {"negative", "neutral"}
    mostly_neutral = sorted(airline for airline, label in largest.items() if label == "neutral")
    assert mostly_neutral == ["Delta", "Virgin America"]
    assert "neutral is the largest class among Delta's and Virgin America's test tweets" in text


def test_replay_table_matches_the_snapshot_meta():
    text = section("Replay of the test set through the stack", level=3)
    assert f"recorded on {SNAPSHOT['recorded_at_utc'][:10]} with `producer --speedup {SNAPSHOT['speedup']}`" in text
    assert table(text) == [
        ["Tweets sent", f"{SNAPSHOT['tweets_sent']:,}"],
        ["Predictions stored in MongoDB", f"{SNAPSHOT['records']['predictions']:,}"],
        ["Airline-hour windows", str(SNAPSHOT["records"]["airline_windows"])],
        ["Sum of the window totals", f"{SNAPSHOT['window_total']:,}"],
        ["Rows dropped by the watermark", str(SNAPSHOT["rows_dropped_by_watermark"])],
    ]


def test_consistency_table_matches_the_report():
    counts = CONSISTENCY["stream_label_counts"]
    assert table(section("Stream predictions against the offline model", level=3)) == [
        ["Tweets in the recorded run", f"{CONSISTENCY['tweets']:,}"],
        ["Same label from the stream and from the offline model", f"{CONSISTENCY['same_prediction_as_offline_model']:,}"],
        ["Accuracy of the stream's labels", score(CONSISTENCY["stream_accuracy"])],
        ["Macro-F1 of the stream's labels", score(CONSISTENCY["stream_macro_f1"])],
        ["Labels assigned by the stream", ", ".join(f"{counts[label]:,} {label}" for label in counts)],
    ]


def test_benchmark_section_is_a_single_sentence_until_the_report_exists():
    if BENCHMARK_EXISTS:
        pytest.skip("reports/benchmark.json exists: its own tests cover the Benchmark section")
    text = section("Benchmark", level=3)
    assert text == BENCHMARK_PLACEHOLDER
    assert not re.search(r"\d", text)
    assert os.path.isfile(os.path.join(ROOT, ".github", "workflows", "benchmark.yml"))
    summary = section("Problem → Approach → Results")
    assert "Throughput and latency are not reported yet" in summary


# --- how it works, how to run, limitations -------------------------------------------------------


def test_diagram_agrees_with_the_configuration():
    blocks = re.findall(r"```mermaid\n(.*?)```", README, flags=re.DOTALL)
    assert len(blocks) == 1 and blocks[0].startswith("flowchart")
    diagram = blocks[0]
    stream = load_config()["stream"]
    partitions = re.search(r"--partitions (\d+)", read("docker-compose.yml")).group(1)
    assert f"{partitions} partitions, key = tweet_id" in diagram
    assert stream["watermark"] == "30 minutes" and "30 min watermark" in diagram
    assert stream["window"] == "1 hour" and "1-hour windows" in diagram
    assert "pandas_udf" in diagram and "dedupe" in diagram
    mongodb = stream["mongodb"]
    assert f"{mongodb['predictions_collection']}, {mongodb['windows_collection']}" in diagram
    for node in ("Replay producer", "Kafka", "Spark Structured Streaming", "MongoDB", "Streamlit Cloud"):
        assert node in diagram
    assert stream["kafka"]["topic"] in read("docker-compose.yml")


def test_tech_stack_has_at_most_six_entries_and_idempotency_is_explained():
    how = section("How it works")
    stack = how[how.index("Tech stack:") :].split("\n\n")[1]
    assert 1 <= len([line for line in stack.splitlines() if line.startswith("- ")]) <= 6
    for phrase in ("`_id = tweet_id`", "`_id = airline|window_start`", "`dropDuplicatesWithinWatermark`"):
        assert phrase in how
    assert "dropDuplicatesWithinWatermark" in read("src", "stream", "transform.py")
    assert '"operationType", "replace"' in read("src", "stream", "processor.py")


def test_how_to_run_has_at_most_four_commands_and_states_the_python_version():
    how = section("How to run")
    commands = re.search(r"```bash\n(.*?)```", how, flags=re.DOTALL).group(1).strip().splitlines()
    assert commands == [
        "docker compose up -d",
        "docker compose run --rm producer --speedup 600",
        "# open http://localhost:8501",
        "pytest -q",
    ]
    assert "Requires Python 3.11 (Spark 3.5)" in how
    ci = yaml.safe_load(read(".github", "workflows", "ci.yml"))
    steps = ci["jobs"]["test"]["steps"]
    assert next(s for s in steps if "setup-python" in s.get("uses", ""))["with"]["python-version"] == "3.11"
    assert re.search(r"^pyspark==3\.5\.", read("requirements-spark.txt"), flags=re.MULTILINE)
    assert "8501:8501" in read("docker-compose.yml")
    assert str(SNAPSHOT["speedup"]) in commands[1]


def test_limitations_agree_with_the_data_and_the_configuration():
    text = section("Limitations")
    split, cfg = METRICS["split"], load_config()

    def utc_days(first: str, last: str) -> int:
        return (pd.Timestamp(last).normalize() - pd.Timestamp(first).normalize()).days + 1

    assert utc_days(split["train"]["first_tweet"], split["test"]["last_tweet"]) == 8
    assert utc_days(split["test"]["first_tweet"], split["test"]["last_tweet"]) == 2
    assert "from 2015 and covers 8 days; the test set is the last 2 of them" in text
    assert split["train"]["first_tweet"].startswith("2015")

    f1 = METRICS["test"]["model"]["f1_per_class"]
    assert min(f1, key=f1.get) == "neutral" and f"its F1 is {score(f1['neutral'])}" in text

    by_airline = {a["airline"]: a["n"] for a in METRICS["test"]["by_airline"]}
    assert max(by_airline, key=by_airline.get) == "American"
    assert f"({by_airline['American']:,} of {split['test']['rows']:,} tweets)" in text
    train, test, _ = prepare(load_raw(), cfg)
    share = lambda part: (part["airline"] == "American").mean()  # noqa: E731
    assert share(train) < share(test) / 2  # "under-represented in the training days"

    assert cfg["stream"]["watermark"] == "30 minutes" and "30-minute watermark" in text
    assert "`spark.sql.shuffle.partitions`" in text and "`docker compose down -v`" in text
    assert "CC BY-NC-SA 4.0" in text

    min_tweets = cfg["dashboard"]["min_tweets"]
    assert f"at least {min_tweets} tweets" in text and f"reach {min_tweets} tweets in an hour" in text
    # "Only the airlines with many tweets reach 20 tweets in an hour": in the recorded run the two
    # busiest airlines hold most of the windows that do, and the quietest airline holds none.
    windows = pd.read_parquet(os.path.join(ROOT, "reports", "stream_snapshot", "windows.parquet"))
    busy = windows[windows["total"] >= min_tweets].groupby("airline").size()
    tweets = windows.groupby("airline")["total"].sum().sort_values(ascending=False)
    assert busy.reindex(tweets.index[:2]).sum() > busy.sum() / 2
    assert busy.get(tweets.index[-1], 0) == 0


def test_license_section_names_both_licenses():
    text = section("Authors and license")
    assert "MIT" in text and "CC BY-NC-SA 4.0" in text
    assert "Phan Thanh Tan" in text and "Phan Thanh Tan" in read("LICENSE")
    for path in ("LICENSE", "data/raw/SOURCE.md"):
        assert f"]({path})" in text and os.path.isfile(os.path.join(ROOT, path))


# --- benchmark (reports/benchmark.json, measured on a GitHub Actions runner) -----------------------


@pytest.fixture(scope="module")
def benchmark():
    if not BENCHMARK_EXISTS:
        pytest.skip("reports/benchmark.json has not been added yet")
    return load_json("reports", "benchmark.json")


def tables(text: str) -> list[list[list[str]]]:
    """Every Markdown table in `text`, each as its body rows."""
    found, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            found.append(current[2:])
            current = []
    return found


def seconds(value: float) -> str:
    return f"{value:.3f} s"


def milliseconds(value: float) -> str:
    return f"{value:.1f} ms"


def test_benchmark_report_comes_from_a_github_runner_and_passed_its_checks(benchmark):
    stream, micro = benchmark["streaming"], benchmark["udf_vs_pandas_udf"]
    assert stream["machine"]["github_runner"] is True and micro["machine"]["github_runner"] is True
    assert stream["commit"] == micro["commit"] and re.fullmatch(r"[0-9a-f]{40}", stream["commit"])
    checks = stream["checks"]
    assert checks["passed"] is True
    assert checks["rows_dropped_by_watermark"] == 0
    assert checks["window_total_added"] == checks["timed_tweets_sent"] == stream["throughput"]["records"]


def test_benchmark_section_names_its_source_and_machine(benchmark):
    text = section("Benchmark", level=3)
    stream = benchmark["streaming"]
    machine, docker, settings = stream["machine"], stream["docker"], stream["settings"]
    assert text.startswith("Source: `reports/benchmark.json`, measured by `.github/workflows/benchmark.yml`")
    for script in ("scripts/benchmark_stream.py", "scripts/benchmark_udf.py"):
        assert f"`{script}`" in text and os.path.isfile(os.path.join(ROOT, script))
    assert f"({stream['measured_at_utc'][:10]}, commit `{stream['commit'][:7]}`)" in text
    assert (
        f"Machine: one GitHub Actions runner, {machine['nproc']} vCPU, {machine['ram_total_gb']} GB RAM, "
        f"{machine['cpu_model']}, Docker {docker['docker_version']}; trigger {settings['trigger']}, "
        f"{settings['topic_partitions']} topic partitions, "
        f"Spark limited to {docker['container_memory_limits_mb']['processor']} MB."
    ) in text
    load = stream["load"]
    assert (
        f"replayed {load['rounds']} times at maximum rate: {stream['throughput']['records']:,} tweets "
        f"sent in {seconds(load['produce_seconds'])}"
    ) in text
    assert load["rounds"] * load["records_per_round"] == stream["throughput"]["records"]


def test_benchmark_stream_table_matches_the_report(benchmark):
    stream = benchmark["streaming"]
    throughput, latency = stream["throughput"], stream["latency_seconds"]

    def percentiles(values: dict) -> str:
        return " / ".join(seconds(values[p]) for p in ("p50", "p95", "p99"))

    assert tables(section("Benchmark", level=3))[0] == [
        ["Tweets", f"{throughput['records']:,}"],
        ["First send to last write", seconds(throughput["first_send_to_last_write_seconds"])],
        ["Throughput", f"{throughput['records_per_second']} tweets per second"],
        ["Latency from send to `written_at`: p50 / p95 / p99", percentiles(latency["to_written_at"])],
        [
            "Latency from send to MongoDB acknowledgement: p50 / p95 / p99",
            percentiles(latency["to_mongodb_acknowledgement"]),
        ],
        ["Rows dropped by the watermark", str(stream["checks"]["rows_dropped_by_watermark"])],
    ]


def test_benchmark_batch_table_matches_the_report(benchmark):
    stream = benchmark["streaming"]

    def row(name: str) -> list[str]:
        query = stream[f"{name}_query"]
        steps = query["duration_ms"]
        return [
            name,
            str(query["batches"]),
            *(
                f"{milliseconds(steps[step]['median_ms'])} / {milliseconds(steps[step]['p95_ms'])}"
                for step in ("addBatch", "triggerExecution")
            ),
        ]

    text = section("Benchmark", level=3)
    assert tables(text)[1] == [row("predictions"), row("airline_windows")]
    assert f"Only {stream['predictions_query']['batches']} micro-batches carried the load" in text


def test_benchmark_udf_table_and_its_conclusion_match_the_report(benchmark):
    micro = benchmark["udf_vs_pandas_udf"]
    text = section("Benchmark", level=3)

    def row(label: str, name: str) -> list[str]:
        return [
            label,
            ", ".join(seconds(run) for run in micro["run_seconds"][name]),
            seconds(micro["median_seconds"][name]),
            f"{micro['median_rows_per_second'][name]:,}",
        ]

    assert tables(text)[2] == [
        row("`udf`: one model call per row", "udf"),
        row("`pandas_udf`: one model call per Arrow batch", "pandas_udf"),
    ]
    assert f"scores {micro['rows']:,} rows in {micro['partitions']} partitions" in text
    assert micro["timed_runs_per_variant"] == 3 and "three timed runs after a warm-up" in text
    # The sentence must say which variant was faster, whichever way the measurement went.
    ratio = micro["pandas_udf_speedup_over_udf"]
    assert ratio == round(micro["median_seconds"]["udf"] / micro["median_seconds"]["pandas_udf"], 2)
    if ratio >= 1:
        conclusion = f"The median `pandas_udf` run took {ratio} times less time than the median `udf` run."
    else:
        conclusion = f"The median `pandas_udf` run took {round(1 / ratio, 2)} times longer than the median `udf` run."
    assert conclusion in text


def test_speed_line_of_the_summary_matches_the_report(benchmark):
    stream = benchmark["streaming"]
    machine = stream["machine"]
    assert (
        f"handled {stream['throughput']['records_per_second']} tweets per second with a median latency of "
        f"{seconds(stream['latency_seconds']['to_written_at']['p50'])} from send to `written_at`, on a GitHub "
        f"Actions runner ({machine['nproc']} vCPU, {machine['ram_total_gb']} GB RAM)"
    ) in section("Problem → Approach → Results")


def test_benchmark_limitation_matches_the_report(benchmark):
    stream = benchmark["streaming"]
    query = stream["predictions_query"]
    median = query["duration_ms"]["triggerExecution"]["median_ms"]
    trigger_seconds = pd.Timedelta(stream["settings"]["trigger"]).total_seconds()
    text = section("Limitations")
    assert f"a single {stream['machine']['nproc']}-vCPU GitHub Actions runner with nothing else running" in text
    assert (
        f"Its {query['batches']} micro-batches took a median of {milliseconds(median)}, "
        f"longer than the {trigger_seconds:.0f}-second trigger"
    ) in text
    # The README says the batches outlast the trigger; that has to be what was measured.
    assert median > trigger_seconds * 1000
    assert "says nothing about behaviour under concurrent load" in text
