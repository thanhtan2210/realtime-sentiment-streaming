import logging
import sys


def get_logger(name: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # These two log every connection and every Spark callback at INFO.
    logging.getLogger("kafka").setLevel(logging.WARNING)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    return logging.getLogger(name)
