# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import gzip

import pandas as pd

_LOGGER = logging.getLogger(__name__)


def load_cell_labels(fpath: str, sep: str = '\t') -> pd.Series:
    """Load cell labels from plain-text file."""

    cell_labels = pd.read_csv(fpath, sep=sep, index_col=0, squeeze=True)

    if sep == '\t':
        delimited_str = 'tab-delimited'
    elif sep == ',':
        delimited_str = 'comma-delimited'
    else:
        delimited_str = '"%s"-delimited' % sep
    _LOGGER.info('Loaded labels for %d cells from %s plain-text file.',
                 cell_labels.size, delimited_str)

    return cell_labels


def save_cell_labels(cell_labels: pd.Series, fpath: str,
                     sep: str = '\t') -> None:
    """Save cell labels to plain-text file."""

    if sep == '\t':
        delimited_str = 'tab-delimited'
    elif sep == ',':
        delimited_str = 'comma-delimited'
    else:
        delimited_str = '"%s"-delimited' % sep
    cell_labels.to_csv(fpath, sep='\t')
    _LOGGER.info('Saved labels for %d cells to %s plain-text file.',
                 cell_labels.size, delimited_str)


def is_gzip_file(fpath: str) -> bool:
    is_gzip = True
    with gzip.open(fpath, 'rb') as fh:
        try:
            fh.read(1)
        except OSError:
            is_gzip = False

    return is_gzip
