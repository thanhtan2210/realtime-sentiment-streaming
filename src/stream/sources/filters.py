"""Filters for English Bluesky posts, shared by the live producer and scripts/probe_bluesky.py.

The keywords, official accounts and phrases live in config/live_sources.yaml.
"""
import os
import re
import socket
import struct
import time
from datetime import datetime, timezone

import yaml

from src.common import config

CONFIG_PATH = os.path.join(config.BASE_DIR, "config", "live_sources.yaml")
MENTION = "app.bsky.richtext.facet#mention"


def load_live_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def phrase_pattern(phrase: str) -> str:
    """A phrase as a whole-word regex: not preceded by a word character or "@", not followed by a word character."""
    return r"(?<![\w@])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"


class PhraseMatcher:
    """Case-insensitive whole-word matching of a list of phrases."""

    def __init__(self, phrases: list[str]):
        self.phrases = list(phrases)
        self.patterns = [re.compile(phrase_pattern(phrase), flags=re.IGNORECASE) for phrase in self.phrases]

    def first_match(self, text: str) -> str | None:
        return next((phrase for phrase, pattern in zip(self.phrases, self.patterns) if pattern.search(text)), None)


def official_accounts(airline_cfg: dict) -> list[dict]:
    account = airline_cfg.get("official_account")
    return [account] if isinstance(account, dict) else []


class AirlineMatcher:
    """Which airlines a post names: by keyword, by "@official.handle", or by a mention facet of the official DID."""

    def __init__(self, airlines_cfg: dict[str, dict]):
        self.keywords = {
            airline: PhraseMatcher(
                spec["keywords"] + ["@" + account["handle"] for account in official_accounts(spec)]
            )
            for airline, spec in airlines_cfg.items()
        }
        self.dids = {
            account["did"]: airline for airline, spec in airlines_cfg.items() for account in official_accounts(spec)
        }

    @property
    def airlines(self) -> list[str]:
        return list(self.keywords)

    def airlines_in(self, text: str, mentioned_dids=()) -> list[str]:
        by_did = {self.dids[did] for did in mentioned_dids if did in self.dids}
        return [
            airline
            for airline, matcher in self.keywords.items()
            if airline in by_did or matcher.first_match(text) is not None
        ]

    def via_did_only(self, text: str, mentioned_dids=()) -> list[str]:
        """Airlines found through a mention facet but not through any keyword."""
        return [
            self.dids[did]
            for did in mentioned_dids
            if did in self.dids and self.keywords[self.dids[did]].first_match(text) is None
        ]


def mentioned_dids(record: dict) -> list[str]:
    """DIDs of the accounts a post mentions, from its rich-text facets."""
    dids = []
    for facet in record.get("facets") or []:
        for feature in (facet or {}).get("features") or []:
            if isinstance(feature, dict) and feature.get("$type") == MENTION and isinstance(feature.get("did"), str):
                dids.append(feature["did"])
    return dids


def is_language(langs, code: str) -> bool:
    return any(lang == code or lang.startswith(code + "-") for lang in langs or [] if isinstance(lang, str))


def parse_created_at(value) -> float | None:
    """createdAt in seconds since the epoch; None if missing or unreadable (it is written by clients)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def clock_offset(server: str, timeout: float = 5.0) -> float | None:
    """Seconds to add to this machine's clock to get the time server's (SNTP); None if unreachable."""
    ntp_to_unix = 2208988800
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.settimeout(timeout)
            sent = time.time()
            udp.sendto(b"\x1b" + 47 * b"\0", (server, 123))
            data, _ = udp.recvfrom(48)
            back = time.time()
    except OSError:
        return None

    def timestamp(offset: int) -> float:
        seconds, fraction = struct.unpack("!II", data[offset : offset + 8])
        return seconds - ntp_to_unix + fraction / 2**32

    received_by_server, sent_by_server = timestamp(32), timestamp(40)
    return round(((received_by_server - sent) + (sent_by_server - back)) / 2, 3)


HANDLE = re.compile(r"(?<![\w@])@[\w.-]+")


def mask_handles(text: str, keep: set[str]) -> str:
    """Replace other people's @handles with "@user"; keep the airline handles listed in `keep`."""

    def replace(match: re.Match) -> str:
        handle = match.group(0)
        if any(handle.lower() == kept or handle.lower().startswith(kept + ".") for kept in keep):
            return handle
        return "@user"

    return HANDLE.sub(replace, text)
