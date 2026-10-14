"""The Bluesky probe's keyword filter and counters, without any network."""
import json

import pytest

from scripts.probe_bluesky import AirlineMatcher, ProbeStats, append_run, is_language, load_live_config


@pytest.fixture(scope="module")
def matcher():
    return AirlineMatcher(load_live_config()["airlines"])


@pytest.mark.parametrize(
    "text",
    [
        "Sunset over the Mississippi delta tonight",
        "Born and raised in the United States",
        "Reading about American history",
        "The Delta variant is back",
        "We united against the plan",
        "southwest of the city",
        "Alaska is beautiful in October",
        "Our united flightplan for the robot",  # "flightplan" is not "flight"
        "email me at someone@united.example",  # an e-mail address, not a handle
        "@unitedairlinesfan posted again",  # a different handle that only starts like one
    ],
)
def test_ordinary_uses_of_the_words_do_not_match(matcher, text):
    assert matcher.airlines_in(text) == []


@pytest.mark.parametrize(
    "text, airlines",
    [
        ("Stuck on a United Airlines plane again", ["United"]),
        ("@united lost my bag", ["United"]),
        ("Thanks @United.", ["United"]),
        ("my united   flight is late", ["United"]),
        ("AMERICAN AIRLINES gate change", ["American"]),
        ("@AmericanAir please help", ["American"]),
        ("booked an AA flight to Dallas", ["American"]),
        ("Delta Air Lines on time today", ["Delta"]),
        ("delta airlines and @delta both", ["Delta"]),
        ("my Delta flight", ["Delta"]),
        ("Southwest Airlines boarding", ["Southwest"]),
        ("@SouthwestAir thanks!", ["Southwest"]),
        ("JetBlue wifi is great", ["JetBlue"]),
        ("Alaska Airlines and @AlaskaAir", ["Alaska"]),
        ("@united vs @delta today", ["United", "Delta"]),
    ],
)
def test_airline_mentions_match(matcher, text, airlines):
    assert matcher.airlines_in(text) == airlines


def test_language_check_accepts_regional_codes():
    assert is_language(["en"], "en") and is_language(["pt", "en-US"], "en")
    assert not is_language(["eng"], "en") and not is_language(None, "en") and not is_language([], "en")


def event(text, langs=("en",), operation="create", created_at="2026-10-14T04:00:00Z", time_us=1_791_950_400_000_000):
    return {
        "kind": "commit",
        "time_us": time_us,
        "commit": {"operation": operation, "record": {"text": text, "langs": list(langs), "createdAt": created_at}},
    }


def test_stats_count_only_new_english_posts_and_store_no_text(matcher, tmp_path):
    stats = ProbeStats(list(matcher.patterns), "en", tolerance_seconds=300)
    received = 1_791_950_402.0  # two seconds after both timestamps
    assert stats.add(event("@united lost my bag"), received, matcher) == ["United"]
    stats.add(event("@united lost my bag", operation="delete"), received, matcher)
    stats.add(event("@united perdió mi maleta", langs=("es",)), received, matcher)
    stats.add(event("nothing to see", created_at="2026-10-14T03:50:00Z"), received, matcher)  # 10 minutes off
    stats.add({"kind": "identity"}, received, matcher)

    report = stats.report(minutes=30)
    assert (report["events"], report["create_posts"], report["english_posts"]) == (5, 3, 2)
    assert report["matched_posts_by_airline"]["United"] == 1 and report["matched_posts"] == 1
    assert report["matched_posts_per_hour"] == 2.0  # one post in half an hour
    assert report["created_at_more_than_tolerance_from_time_us"] == 1
    assert report["source_lag"]["receive_minus_jetstream_time_us"]["median_seconds"] == 2.0
    assert report["source_lag"]["matched_receive_minus_created_at"] == {"n": 1, "median_seconds": 2.0, "p95_seconds": 2.0}

    path = tmp_path / "probe.json"
    append_run({"run": 1, **report}, str(path))
    append_run({"run": 2, **report}, str(path))
    written = path.read_text(encoding="utf-8")
    assert [run["run"] for run in json.loads(written)["runs"]] == [1, 2]
    for fragment in ("lost my bag", "maleta", "nothing to see", "did:"):
        assert fragment not in written
