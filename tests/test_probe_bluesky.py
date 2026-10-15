"""The Bluesky probe's filters, counters and review sample, without any network."""
import csv
import json
import os

import pytest

from scripts.probe_bluesky import (
    UNSPECIFIED,
    AirlineMatcher,
    PhraseMatcher,
    ProbeStats,
    ReservoirSample,
    append_run,
    is_language,
    load_live_config,
    main,
    mask_handles,
    mentioned_dids,
    runner_info,
    write_sample,
)

CFG = load_live_config()
UNITED_DID = CFG["airline_strict"]["United"]["official_account"]["did"]


@pytest.fixture(scope="module")
def airlines():
    return AirlineMatcher(CFG["airline_strict"])


@pytest.fixture(scope="module")
def travel():
    return PhraseMatcher(CFG["air_travel"])


# --- config --------------------------------------------------------------------------------


def test_every_airline_states_its_official_account_or_none_found():
    for airline, spec in CFG["airline_strict"].items():
        account = spec["official_account"]
        assert account == "none found" or (
            set(account) == {"handle", "did"} and account["did"].startswith("did:plc:")
        ), airline
    accepted = {a: s["official_account"]["handle"] for a, s in CFG["airline_strict"].items() if s["official_account"] != "none found"}
    assert accepted == {"United": "united.com", "Southwest": "southwest.com"}


# --- airline_strict --------------------------------------------------------------------------


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
        "Our united flightplan for the robot",
        "email me at someone@united.example",
        "@unitedairlinesfan posted again",
    ],
)
def test_ordinary_uses_of_airline_words_do_not_match(airlines, text):
    assert airlines.airlines_in(text) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Stuck on a United Airlines plane again", ["United"]),
        ("@united lost my bag", ["United"]),
        ("Thanks @united.com for nothing", ["United"]),
        ("my united   flight is late", ["United"]),
        ("AMERICAN AIRLINES gate change", ["American"]),
        ("@AmericanAir please help", ["American"]),
        ("booked an AA flight to Dallas", ["American"]),
        ("Delta Air Lines on time today", ["Delta"]),
        ("my Delta flight", ["Delta"]),
        ("@southwest.com where is my bag", ["Southwest"]),
        ("@SouthwestAir thanks!", ["Southwest"]),
        ("JetBlue wifi is great", ["JetBlue"]),
        ("Alaska Airlines and @AlaskaAir", ["Alaska"]),
        ("@united vs @delta today", ["United", "Delta"]),
    ],
)
def test_airline_mentions_match(airlines, text, expected):
    assert airlines.airlines_in(text) == expected


def test_a_mention_facet_of_an_official_account_counts_even_if_the_text_does_not_say_it(airlines):
    record = {
        "text": "thanks for the help today",
        "facets": [
            {"features": [{"$type": "app.bsky.richtext.facet#link", "uri": "https://example.com"}]},
            {"features": [{"$type": "app.bsky.richtext.facet#mention", "did": UNITED_DID}]},
            {"features": [{"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:someoneelse"}]},
        ],
    }
    dids = mentioned_dids(record)
    assert dids == [UNITED_DID, "did:plc:someoneelse"]
    assert airlines.airlines_in(record["text"], dids) == ["United"]
    assert airlines.via_did_only(record["text"], dids) == ["United"]
    assert airlines.via_did_only("@united.com thanks", dids) == []  # the text already names it
    assert mentioned_dids({"text": "no facets"}) == [] and mentioned_dids({"facets": [None, {}]}) == []


# --- air_travel ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the Mississippi delta",
        "United States",
        "a flight of stairs to the attic",
        "new flight simulator build",
        "my flighty cat",
        "the bag was checked twice",
        "no layovers here",  # "layovers" is not the phrase "layover"
        "tarmac repaving downtown",
    ],
)
def test_ordinary_text_does_not_match_air_travel(travel, text):
    assert travel.first_match(text) is None


@pytest.mark.parametrize(
    "text, phrase",
    [
        ("My flight was delayed three hours", "flight was delayed"),
        ("FLIGHT CANCELLED again", "flight cancelled"),
        ("they canceled... flight canceled", "flight canceled"),
        ("sitting on the tarmac for 2 hours", "on the tarmac"),
        ("TSA PreCheck line was empty", "tsa precheck"),
        ("4 hour layover in Denver", "layover"),
        ("the gate agent was lovely", "gate agent"),
        ("lost my luggage in Paris", "lost my luggage"),
    ],
)
def test_air_travel_phrases_match(travel, text, phrase):
    assert travel.first_match(text) == phrase


# --- counting -------------------------------------------------------------------------------


def event(text, langs=("en",), operation="create", created_at="2026-10-14T13:00:00Z",
          time_us=1_791_982_800_000_000, facets=None):  # fmt: skip
    record = {"text": text, "langs": list(langs), "createdAt": created_at}
    if facets:
        record["facets"] = facets
    return {"kind": "commit", "time_us": time_us, "commit": {"operation": operation, "record": record}}


def test_stats_split_the_two_levels_and_tag_unspecified(airlines, travel):
    stats = ProbeStats(airlines.airlines, travel.phrases, "en", tolerance_seconds=300)
    received = 1_791_982_802.0  # two seconds after both timestamps
    assert stats.add(event("@united my flight was delayed"), received, airlines, travel) == (["United"], "flight was delayed")
    assert stats.add(event("missed my flight, sigh"), received, airlines, travel) == ([], "missed my flight")
    mention = [{"features": [{"$type": "app.bsky.richtext.facet#mention", "did": UNITED_DID}]}]
    assert stats.add(event("thanks for the help", facets=mention), received, airlines, travel) == (["United"], None)
    assert stats.add(event("my flight", langs=("de",)), received, airlines, travel) is None
    stats.add(event("nothing here", created_at="2026-10-14T12:50:00Z"), received, airlines, travel)  # 10 min off
    stats.add({"kind": "identity"}, received, airlines, travel)

    report = stats.report(minutes=30, offset=0.5)
    assert (report["events"], report["create_posts"], report["english_posts"]) == (6, 5, 4)
    assert report["english_posts_per_hour"] == 8.0
    strict = report["airline_strict"]
    assert strict["matched_posts"] == 2 and strict["matched_posts_by_airline"]["United"] == 2
    assert strict["matched_only_through_a_mention_facet"] == 1
    assert strict["matched_posts_per_hour"] == 4.0
    air = report["air_travel"]
    assert air["matched_posts"] == 2 and air["also_airline_strict"] == 1
    assert air["matched_posts_by_airline_or_unspecified"]["United"] == 1
    assert air["matched_posts_by_airline_or_unspecified"][UNSPECIFIED] == 1
    assert air["matched_posts_by_first_phrase"]["missed my flight"] == 1
    assert report["created_at_more_than_tolerance_from_time_us"] == 1
    lag = report["source_lag"]
    assert lag["clock_offset_applied_seconds"] == 0.5
    assert lag["receive_minus_jetstream_time_us"]["median_seconds"] == 2.0
    assert lag["receive_minus_jetstream_time_us_corrected"]["median_seconds"] == 2.5


def test_language_check_accepts_regional_codes():
    assert is_language(["en"], "en") and is_language(["pt", "en-US"], "en")
    assert not is_language(["eng"], "en") and not is_language(None, "en") and not is_language([], "en")


# --- what is written ------------------------------------------------------------------------


def test_report_holds_no_text_handle_or_did(airlines, travel, tmp_path):
    stats = ProbeStats(airlines.airlines, travel.phrases, "en", tolerance_seconds=300)
    mention = [{"features": [{"$type": "app.bsky.richtext.facet#mention", "did": UNITED_DID}]}]
    stats.add(event("@united.com my flight was delayed @alice.bsky.social", facets=mention), 1_791_982_802.0, airlines, travel)
    path = tmp_path / "probe.json"
    append_run({"run": 1, **stats.report(30, None)}, str(path))
    append_run({"run": 2, **stats.report(30, None)}, str(path))
    written = path.read_text(encoding="utf-8")
    assert [run["run"] for run in json.loads(written)["runs"]] == [1, 2]
    for fragment in ("was delayed @", "alice", "did:plc:"):
        assert fragment not in written


def test_handles_are_masked_in_the_review_sample_except_airline_ones():
    keep = {"@united", "@southwest.com"}
    text = "@alice.bsky.social and @united.com and @southwest.com and @bob said hi; mail x@y.com"
    assert mask_handles(text, keep) == "@user and @united.com and @southwest.com and @user said hi; mail x@y.com"


def test_reservoir_sample_is_uniform_sized_and_repeatable():
    def run():
        sample = ReservoirSample(size=40, seed=42)
        for i in range(1000):
            sample.offer(i)
        return sample

    first, second = run(), run()
    assert len(first.items) == 40 and first.seen == 1000
    assert first.items == second.items
    assert max(first.items) > 40  # later items get in, not just the first 40
    small = ReservoirSample(size=40, seed=42)
    for i in range(5):
        small.offer(i)
    assert small.items == [0, 1, 2, 3, 4]


def test_review_csv_has_three_columns_and_no_handle(tmp_path):
    sample = ReservoirSample(size=40, seed=42)
    sample.offer({"text": "@user my flight was delayed", "matched_phrase": "flight was delayed"})
    path = tmp_path / "review" / "sample.csv"
    write_sample(sample, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["text", "matched_phrase", "relevant"]
    assert rows == [{"text": "@user my flight was delayed", "matched_phrase": "flight was delayed", "relevant": ""}]


def test_the_review_sample_may_not_be_written_inside_the_repository(monkeypatch):
    from src.common import config

    inside = os.path.join(config.BASE_DIR, "reports", "sample.csv")
    monkeypatch.setattr("sys.argv", ["probe_bluesky", "--minutes", "0", "--sample-csv", inside])
    with pytest.raises(SystemExit):
        main()


def test_a_sample_path_outside_the_repository_is_accepted(monkeypatch, tmp_path):
    """Like $RUNNER_TEMP on GitHub Actions: outside the checkout, so allowed."""
    calls = []
    monkeypatch.setattr("scripts.probe_bluesky.asyncio.run", lambda coro: calls.append(coro.close()) or {})
    monkeypatch.setattr("scripts.probe_bluesky.clock_offset", lambda server: None)
    output, sample = tmp_path / "probe.json", tmp_path / "runner_temp" / "bluesky_sample.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["probe_bluesky", "--minutes", "0", "--output", str(output), "--sample-csv", str(sample)],
    )
    assert main() == 0
    run = json.loads(output.read_text(encoding="utf-8"))["runs"][0]
    assert run["runner"] in {"local", "github-actions"} and run["os"]
    assert sample.exists()


def test_runner_info_tells_github_actions_from_this_machine():
    assert runner_info({"GITHUB_ACTIONS": "true", "ImageOS": "ubuntu24"})["runner"] == "github-actions"
    assert runner_info({"GITHUB_ACTIONS": "true", "ImageOS": "ubuntu24"})["github_runner_image"] == "ubuntu24"
    local = runner_info({})
    assert local["runner"] == "local" and local["github_runner_image"] is None and local["os"]
