"""Stream settings: the `stream` section of config/pipeline.yaml plus environment overrides."""
import copy
import os

# environment variable -> path of the key it overrides inside cfg["stream"]
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
}


def stream_settings(cfg: dict, env=os.environ) -> dict:
    settings = copy.deepcopy(cfg["stream"])
    for variable, path in ENV_OVERRIDES.items():
        if env.get(variable):
            node = settings
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = env[variable]
    return settings
