from __future__ import absolute_import
from __future__ import division

import logging
import os

from qtrader.framework import LOG_NAME, LOG_LEVEL

_level_map = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}

logger = logging.getLogger(LOG_NAME)
logger.setLevel(_level_map[LOG_LEVEL.upper()])
