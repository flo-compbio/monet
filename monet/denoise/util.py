# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Utility functions for denoising."""

import logging
from math import sqrt
from typing import Tuple
import gc
import sys
import time

from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from scipy.stats import poisson

from ..core import ExpMatrix
from ..latent import PCAModel
from ..latent import CompressedData
from .. import util

_LOGGER = logging.getLogger(__name__)


def aggregate_neighbors(
        matrix: ExpMatrix,
        pc_scores: pd.DataFrame,
        num_neighbors: int) \
        -> Tuple[ExpMatrix, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbor aggregation."""

    ### calculate pairwise distances

    t0_total = time.time()

    # apparently the output of pairwise_distance changes depending on
    # whether array is C-contiguous or not
    Y = np.array(pc_scores.values, order='C', copy=False)

    # bug: np.sqrt() reports "input is invalid" for zeros in np.float32 arrays
    invalid_errstate = 'warn'
    if np.issubdtype(Y.dtype, np.float32):
        invalid_errstate = 'ignore'

    with np.errstate(invalid=invalid_errstate):
        t0 = time.time()
        D = pairwise_distances(Y, n_jobs=1, metric='euclidean')
        t1 = time.time()
    _LOGGER.info('Calculating the pairwise distances took %.1f s.', t1-t0)

    ### sort the distance matrix
    t0 = time.time()
    sort_indices = np.argsort(D, axis=1, kind='quicksort')
    t1 = time.time()
    _LOGGER.info('Sorting the pairwise distance matrix took %.1f s.', t1-t0)

    ### aggregate expression values

    X = np.array(matrix.X, order='C', copy=False)
    num_transcripts = X.sum(axis=1)
    dtype = X.dtype

    A = np.empty(
        X.shape, order='C', dtype=dtype)
    neighbors = np.empty(
        (matrix.num_cells, num_neighbors), dtype=np.int64)
    neighbor_dists = np.empty(
        (matrix.num_cells, num_neighbors), dtype=np.float64)
    cell_sizes = np.empty(
        matrix.num_cells, dtype=np.float64)

    t0 = time.time()
    for i in range(matrix.num_cells):
        ind = sort_indices[i, :num_neighbors]
        A[i, :] = np.sum(X[ind, :], axis=0, dtype=dtype)
        neighbors[i, :] = ind
        neighbor_dists[i, :] = D[i, ind]
        cell_sizes[i] = np.median(num_transcripts[ind])
    t1 = time.time()
    _LOGGER.info('Aggregating the expression values took %.1f s.', t1-t0)

    # convert aggregated matrix from numpy to ExpMatrix object
    agg_matrix = ExpMatrix(A.T, genes=matrix.genes, cells=matrix.cells)

    # convert remaining results from numpy to pandas objects
    cell_sizes = pd.Series(cell_sizes, index=matrix.cells)
    neighbors = pd.DataFrame(neighbors, index=matrix.cells)
    neighbor_dists = pd.DataFrame(neighbor_dists, index=matrix.cells)

    t1_total = time.time()
    _LOGGER.info('Nearest-neighbor aggregation took %.1f s.',
                 t1_total-t0_total)

    return agg_matrix, cell_sizes, neighbors, neighbor_dists


def estimate_num_components(
        matrix: ExpMatrix,
        max_components: int = 100,
        var_fold_thresh: float = 2.0, 
        seed: int = 0) -> Tuple[int, float]:
    """Estimate number of PCs by simulating a pure noise matrix."""

    _LOGGER.info('Estimating the number of PCs with var_fold_thresh=%.2f',
                 var_fold_thresh)

    # calculate mean expression for all genes
    transcript_count = matrix.sum(axis=0).median()
    mean = util.scale(matrix, transcript_count).mean(axis=1)

    # simulate matrix with only technical variance
    np.random.seed(seed)
    S = np.empty(matrix.X.shape, dtype=np.uint32)
    for i in range(matrix.num_cells):
        S[i, :] = poisson.rvs(mean.values)
    random_matrix = ExpMatrix(S.T, genes=matrix.genes, cells=matrix.cells)

    # make sure random PCA model uses the same transcript count
    _LOGGER.info('Performing PCA on pure noise matrix...')
    rand_pca_model = PCAModel(max_components, transcript_count, seed=seed)
    rand_pca_model.fit(random_matrix)
    random_var = rand_pca_model.model_.explained_variance_[0]
    var_thresh = var_fold_thresh * random_var
    _LOGGER.info('Variance threshold: %.3f', var_thresh)

    # apply PCA to real matrix and determine
    # number of PCs above variance threshold
    pca_model = PCAModel(max_components, transcript_count, seed=seed)
    pca_model.fit(matrix)
    explained_variance = pca_model.explained_variance_
    num_components = np.sum(explained_variance >= var_thresh)

    _LOGGER.info('The estimated number of PCs is %d.', num_components)

    return num_components, random_var
