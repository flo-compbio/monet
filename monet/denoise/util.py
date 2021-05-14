# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Utility functions for denoising."""

import logging
from typing import Tuple, Union
import time
from math import ceil

from sklearn.neighbors import NearestNeighbors
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

    t0_total = time.time()

    # apparently the output of pairwise_distance changes depending on
    # whether array is C-contiguous or not
    Y = np.array(pc_scores.values, order='C', copy=False)

    ### constructing the KD-tree
    t0 = time.time()
    neigh = NearestNeighbors(n_neighbors=num_neighbors, algorithm='kd_tree')
    neigh.fit(Y)
    t1 = time.time()
    _LOGGER.info('Constructing the KD-tree took %.1f s.', t1-t0)

    ### aggregate expression values
    X = np.array(matrix.X, order='C', copy=False)
    num_transcripts = X.sum(axis=1)
    dtype = X.dtype

    # this returns each point itself as its own nearest neighbor
    # (behavior is different if kneighbors() is called without Y)
    neighbor_dists, neighbors = neigh.kneighbors(Y, return_distance=True)

    A = np.empty(X.shape, order='C', dtype=dtype)
    cell_sizes = np.empty(matrix.num_cells, dtype=np.float64)

    t0 = time.time()
    for i in range(matrix.num_cells):
        ind = neighbors[i, :]
        A[i, :] = np.sum(X[ind, :], axis=0, dtype=dtype)
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


def determine_num_neighbors(
        matrix: ExpMatrix,
        target_transcript_count: Union[int, float],
        max_frac_neighbors: float) -> int:
    """Determine the number of neighbors to use for kNN-aggregation."""

    transcript_count = matrix.median_transcript_count
    _LOGGER.info('The median transcript count is %.1f.',
                    transcript_count)

    num_neighbors = int(ceil(target_transcript_count / transcript_count))

    max_num_neighbors = \
            max(int(max_frac_neighbors * matrix.num_cells), 1)
    if num_neighbors <= max_num_neighbors:
        _LOGGER.info(
            'Will use num_neighbors=%d for aggregation'
            '(value was determined automatically '
            'based on a target transcript count of %.1f).',
            num_neighbors, float(target_transcript_count))
    else:
        _LOGGER.warning(
            'Will use num_neighbors=%d for aggregation, '
            'to not exceed %.1f %% of the total number of cells. '
            'However, based on a target transcript count of %d, '
            'we should use k=%d. As a result, gene loadings '
            'may be biased towards highly expressed genes.',
            max_num_neighbors, 100*max_frac_neighbors,
            target_transcript_count, num_neighbors)
        num_neighbors = max_num_neighbors

    return num_neighbors


def var_estimate_num_components(
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


def denoise_data(matrix: ExpMatrix, num_components: int, pca_model: PCAModel,
                 scale: bool = True) -> ExpMatrix:
    pc_scores = pca_model.transform(matrix)
    agg_matrix, cell_sizes, _, _ = aggregate_neighbors(
        matrix, pc_scores, pca_model.agg_num_neighbors_)

    pca_model = PCAModel(num_components)
    pc_scores = pca_model.fit_transform(agg_matrix)

    if not scale:
        # do not use inferred cell sizes, instead, keep aggregated cell sizes
        cell_sizes = agg_matrix.sum(axis=0)

    compressed_data = CompressedData(pca_model, pc_scores, cell_sizes)
    denoised_matrix = compressed_data.decompress()

    return denoised_matrix
