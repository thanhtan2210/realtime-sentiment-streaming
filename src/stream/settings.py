"""Stream settings: the `stream` section of config/pipeline.yaml plus environment overrides."""
import copy
import os

# environment variable -> path of the key it overrides inside cfg["stream"]
ENV_OVERRIDES = {
    "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
    "KAFKA_TOPIC": ("kafka", "topic"),
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
