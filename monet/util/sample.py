# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

import logging

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

from ..core import ExpMatrix

_LOGGER = logging.getLogger(__name__)


def sample_cells(cells: pd.Index, num_cells: int,
                 cell_labels: pd.Series = None, seed: int = 0) -> pd.Index:
    """Sample cells and, if applicable, maintain cluster proportions."""

    if num_cells > cells.size:
        raise ValueError(
            'Number of cells to sample (%d) exceeds the total number of '
            'cells (%d).' % (num_cells, cells.size))

    if cell_labels is None:
        cell_labels = pd.Series(index=cells, data=np.zeros(cells.size))

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=num_cells,
        random_state=seed)

    ### convert to numeric cluster labels
    # get a list of clusters in the order that they appear
    unique_clusters = cell_labels.loc[~cell_labels.duplicated()]

    cluster_labels = dict(
        [label, i] for i, label in enumerate(unique_clusters))
    numeric_labels = cell_labels.map(cluster_labels)

    sel_indices = list(
        splitter.split(np.arange(cells.size),
        numeric_labels))[0][0]
    sel_indices.sort()
    sample = cells[sel_indices]

    return sample


def sample_labels(cell_labels: pd.Series, num_cells: int,
                  seed: int = 0) -> pd.Series:
    """Sample cells from a cell label vector, maintaining label proportions."""

    sample = sample_cells(
        cell_labels.index, num_cells, cell_labels, seed=seed)
    sample_cell_labels = cell_labels.loc[sample]

    return sample_cell_labels


def sample_matrix(matrix: ExpMatrix, num_cells: int,
                  cell_labels: pd.Series=None, seed: int = 0):
    """Sample cells from an expression matrix."""

    sample = sample_cells(matrix.cells, num_cells, cell_labels,
                          seed=seed)
    matrix = matrix.loc[:, sample]

    return matrix
