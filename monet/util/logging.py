# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Logging functions."""

import logging
import sys
from contextlib import AbstractContextManager


def configure_logger(name='', level=logging.INFO, showtime=False):
    """Configure a root logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []

    if showtime:
        log_format = '[%(asctime)s] (%(name)s) %(levelname)s: %(message)s'
    else:
        log_format = '(%(name)s) %(levelname)s: %(message)s'
    log_date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, log_date_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class LoggingFilter(AbstractContextManager):
    """Attaches a filter to a handler."""

    def __init__(self, logger: logging.Logger, prefix: str,
                 handler_index: int = 0, allow_root: bool = True):

        self.logger = logger
        self.handler = logger.handlers[handler_index]
        self.prefix = prefix
        if allow_root:
            self._filter = lambda x: x.name == 'root' or \
                    x.name.startswith(prefix)
        else:
            self._filter = lambda x: x.name.startswith(prefix)

    def __enter__(self):
        self.handler.addFilter(self._filter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.removeFilter(self._filter)
