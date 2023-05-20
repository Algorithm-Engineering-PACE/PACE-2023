import logging
import sys

LOG_LEVEL = logging.INFO

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(LOG_LEVEL)

    formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()