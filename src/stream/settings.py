"""Pipeline settings from config/pipeline.yaml plus environment overrides.

`stream` is the replay mode (labelled 2015 tweets); `live` is the live mode
(English Bluesky posts). Both use the same environment variables; a variable
only applies where its key exists in the section.
"""
import copy
import os

# environment variable -> path of the key it overrides inside the section
ENV_OVERRIDES = {
    "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
    "KAFKA_TOPIC": ("kafka", "topic"),
    "MONGODB_URI": ("mongodb", "uri"),
    "MONGODB_DATABASE": ("mongodb", "database"),
    "STREAM_WINDOW": ("window",),
    "STREAM_WATERMARK": ("watermark",),
    "STREAM_TRIGGER": ("trigger",),
    "STREAM_CHECKPOINT_DIR": ("checkpoint_dir",),
    "SPARK_MASTER": ("spark", "master"),
    "SPARK_DRIVER_MEMORY": ("spark", "driver_memory"),
    "SPARK_SHUFFLE_PARTITIONS": ("spark", "shuffle_partitions"),
    "LIVE_CURSOR_PATH": ("source", "cursor_path"),
}
MODES = {"replay": "stream", "live": "live"}


def stream_settings(cfg: dict, env=os.environ, section: str = "stream") -> dict:
    settings = copy.deepcopy(cfg[section])
    for variable, path in ENV_OVERRIDES.items():
        if not env.get(variable):
            continue
        node = settings
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and path[-1] in node:
            node[path[-1]] = env[variable]
    return settings


def mode_settings(cfg: dict, mode: str, env=os.environ) -> dict:
    """Settings of one pipeline mode: "replay" or "live"."""
    if mode not in MODES:
        raise ValueError(f"PIPELINE_MODE must be one of {', '.join(MODES)}, not {mode!r}")
    return stream_settings(cfg, env, MODES[mode])
