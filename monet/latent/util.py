# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import gc
import sys
import time
import logging
from typing import Tuple, Union
from math import sqrt, ceil

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import pairwise_distances

from ..core import ExpMatrix
from . import PCAModel
from . import CompressedData
from .. import util

_LOGGER = logging.getLogger(__name__)


def calculate_binomial_params(capture_eff: float, alpha: float):
    """Calculates the subsampling rates for Batson et al. MCV method."""
    p1 = (1 + alpha - sqrt(pow(1+alpha, 2.0) - 4*alpha*capture_eff)) / (2*alpha)
    p2 = alpha*p1
    return p1, p2


def generate_split(matrix: ExpMatrix, capture_eff: float, alpha: float,
                   seed: int = 0) -> Tuple[ExpMatrix, ExpMatrix]:
    """Split the data into independent subsets for MCV (Batson et al.).""" 
    p1, p2 = calculate_binomial_params(capture_eff, alpha)
    _LOGGER.info('Data will be split into datasets containing '
                 '%.1f%% and %.1f%% of transcripts, respectively.',
                 100*(p1/capture_eff), 100*(p2/capture_eff))
    np.random.seed(seed)
    X = matrix.X
    X1 = np.empty(X.shape, dtype=np.uint32)
    X2 = np.empty(X.shape, dtype=np.uint32)
    for i in range(X.shape[0]):
        x1 = np.random.binomial(X[i, :], p1/capture_eff)
        x_shared = np.random.binomial(x1, p2)
        x2 = X[i, :] - x1 + x_shared
        X1[i, :] = x1
        X2[i, :] = x2

    train_matrix = ExpMatrix(X1.T, genes=matrix.genes, cells=matrix.cells)
    test_matrix = ExpMatrix(X2.T, genes=matrix.genes, cells=matrix.cells)
    _LOGGER.info('Done splitting data!')
    
    return train_matrix, test_matrix


def calculate_poisson_loss(matrix: ExpMatrix, ref_matrix: ExpMatrix,
                           min_mu: float = 0.001) -> float:
    """Calculate the Poisson loss for a matrix, given a reference.
    
    Note: It is the responsibility of the caller to make sure that both
    matrices are scaled to the same transcript count."""
    
    X = np.array(matrix.X, dtype=np.float64, copy=True)
    X_ref = np.array(ref_matrix.X, dtype=np.float64, copy=False)

    # apply threshold
    X[X < min_mu] = min_mu
    
    # calculate mean loss
    mean_loss = np.mean(X - X_ref * np.log(X))
    
    return mean_loss


def cross_validate_split(
        compressed_data: CompressedData, val_matrix: ExpMatrix,
        num_component_values: np.ndarray) -> np.ndarray:
    """Calculate Poisson loss for different numbers of PCs."""

    loss_values = np.zeros(len(num_component_values), dtype=np.float64)

    val_transcript_count = val_matrix.sum(axis=0).median()
    val_matrix = util.scale(val_matrix, val_transcript_count)
    gc.collect()

    for i, num_components in enumerate(num_component_values):
        _LOGGER.info('Testing value %d/%d (%d PCs)...',
                     i+1, loss_values.size, num_components)
        sys.stdout.flush()
        # calculate Poisson loss using MCV here
        restored_matrix = compressed_data.decompress(
            num_components=num_components, apply_scaling=False)

        # scale the restored matrix to match the validation matrix
        restored_matrix = util.scale(restored_matrix, val_transcript_count)

        loss_values[i] = calculate_poisson_loss(restored_matrix, val_matrix)
        gc.collect()
    
    return loss_values


def determine_num_neighbors(
        matrix: ExpMatrix,
        target_transcript_count: Union[int, float],
        max_frac_neighbors: float) -> int:
    """Determine the number of neighbors to use for kNN-aggregation."""

    transcript_count = matrix.median_transcript_count
    _LOGGER.info('The median transcript count is %.1f.',
                    transcript_count)

    num_neighbors = int(ceil(target_transcript_count / 
                                transcript_count))
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
            'we should use k=%d. As a result, gene loadings'
            'may be biased towards highly expressed genes.',
            max_num_neighbors, 100*max_frac_neighbors,
            target_transcript_count, num_neighbors)
        num_neighbors = max_num_neighbors

    return num_neighbors


def aggregate_neighbors(
        matrix: ExpMatrix,
        pc_scores: pd.DataFrame,
        num_neighbors: int) \
        -> ExpMatrix:
    """Nearest-neighbor aggregation."""

    ### calculate pairwise distances
    # apparently the output of `pairwise_distances` changes depending on
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
    A = np.empty(X.shape, order='C', dtype=X.dtype)
    t0 = time.time()
    for i in range(matrix.num_cells):
        ind = sort_indices[i, :num_neighbors]
        A[i, :] = np.sum(X[ind, :], axis=0, dtype=X.dtype)
    t1 = time.time()
    _LOGGER.info('Aggregating the expression values took %.1f s.', t1-t0)

    # convert aggregated matrix from numpy to ExpMatrix object
    agg_matrix = ExpMatrix(A.T, genes=matrix.genes, cells=matrix.cells)

    return agg_matrix
