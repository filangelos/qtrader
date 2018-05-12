import logging
import os

from qtrader.framework import LOG_NAME, LOG_LEVEL

_level_map = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}

logging.basicConfig(level=_level_map[LOG_LEVEL.upper()])

logger = logging.getLogger(LOG_NAME)
