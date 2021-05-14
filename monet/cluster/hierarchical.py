# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import time
import sys

from ..core import ExpMatrix
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

_LOGGER = logging.getLogger(__name__)


def get_hierarchical_order(X, method, metric):
    y = pdist(X, metric=metric)
    Z = linkage(y, method=method)
    R = dendrogram(Z, no_plot=True)
    order = np.int32([int(i) for i in R['ivl']])
    return order


def order_matrix(
        matrix: ExpMatrix,
        gene_metric: str = 'correlation',
        cell_metric: str = 'euclidean',
        axis: str = 'both') -> ExpMatrix:

    method = 'average'

    _LOGGER.info('Ordering matrix with %d genes and %d cells using '
                 'hierarchical clustering...',
                 matrix.shape[0], matrix.shape[1]); sys.stdout.flush()

    if axis == 'both' or axis == 'genes':
        t0 = time.time()
        gene_order = get_hierarchical_order(
            matrix.X.T, method=method, metric=gene_metric)

        t1 = time.time()
        _LOGGER.info('Determined gene order in %.1f s.', t1-t0)
    else:
        gene_order = np.arange(matrix.shape[0])

    if axis == 'both' or axis == 'cells':
        t0 = time.time()
        cell_order = get_hierarchical_order(
            matrix.X, method=method, metric=cell_metric)

        t1 = time.time()
        _LOGGER.info('Determined cell order in %.1f s.', t1-t0)
    else:
        cell_order = np.arange(matrix.shape[1])

    matrix = matrix.iloc[gene_order, cell_order]
    return matrix
