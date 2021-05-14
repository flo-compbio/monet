# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020, 2021 Florian Wagner
#
# This file is part of Monet.

import logging
import gc
from typing import Tuple
from math import sqrt

import numpy as np

from ..core import ExpMatrix
from ..latent import PCAModel, CompressedData
from .. import util

_LOGGER = logging.getLogger(__name__)


def mcv_estimate_num_components(
        matrix: ExpMatrix,
        max_num_components: int = 100,
        alpha: float = 1/9, capture_eff: float = 0.05,
        num_splits: int = 5, step_size: int = 10,
        fine_step_size: int = 3,
        use_double_precision: bool = False,
        seed: int = 0) -> Tuple[int, np.ndarray]:
    """Use MCV to determine the optimum number of PCs.

    Implements a grid search strategy to tune the number of PCs.
    """

    _LOGGER.info('Using molecular cross-validation to determine '
                 'the number of PCs...')

    if not np.issubdtype(matrix.values.dtype, np.integer):
        raise ValueError('Matrix data type must be integer!')

    # this is the array to hold the loss values for all splits and # PCs
    mcv_results = np.empty((num_splits, max_num_components), dtype=np.float64)
    mcv_results[:, :] = np.inf

    ## first round: test coarse grid of num_component values
    _LOGGER.info('Testing coarse grid of num_component values...')

    num_component_values = np.arange(
        step_size, max_num_components+1, step_size)

    L = cross_validate_num_components(
        matrix, num_component_values,
        max_num_components,
        alpha, capture_eff, num_splits,
        use_double_precision, seed)

    mcv_results[:, num_component_values-1] = L
    gc.collect()

    mean_loss = np.mean(mcv_results, axis=0)
    coarse_num_components = np.argmin(mean_loss) + 1
    _LOGGER.info('Coarse grid search yielded optimum of %d PCs...',
                    coarse_num_components)

    ## second round: zoom in on range that contains the optimal value
    _LOGGER.info('Testing fine grid of num_component values...')

    # make sure we're not testing values larger than max_num_components
    largest_num_components = coarse_num_components + step_size - \
            fine_step_size
    largest_num_components = \
        min(largest_num_components, max_num_components)

    num_component_values = np.r_[
        np.arange(coarse_num_components - step_size + fine_step_size,
                  coarse_num_components, fine_step_size),
        np.arange(largest_num_components, coarse_num_components,
                  -fine_step_size)[::-1]]

    L = cross_validate_num_components(
        matrix, num_component_values,
        max_num_components,
        alpha, capture_eff, num_splits,
        use_double_precision, seed)

    mcv_results[:, num_component_values-1] = L
    del L; gc.collect()

    mean_loss = np.mean(mcv_results, axis=0)
    fine_num_components = np.argmin(mean_loss) + 1
    _LOGGER.info('After fine grid search, optimal number of PCs is %d...',
                fine_num_components)

    ## final round: test every value
    _LOGGER.info('Testing final grid of num_component values...')

    # figure out which num_component values still need to be tested
    num_component_values = []
    for i in range(fine_num_components-2, -1, -1):
        if np.isfinite(mean_loss[i]):
            break
        num_component_values.append(i+1)
    num_component_values = list(reversed(num_component_values))

    for i in range(fine_num_components, max_num_components):
        if np.isfinite(mean_loss[i]):
            break
        num_component_values.append(i+1)

    num_component_values = np.array(num_component_values, dtype=np.uint64)

    L = cross_validate_num_components(
        matrix, num_component_values,
        max_num_components,
        alpha, capture_eff, num_splits,
        use_double_precision, seed)

    mcv_results[:, num_component_values-1] = L
    del L; gc.collect()

    # determine final value for the optimal number of PCs
    mean_loss = np.mean(mcv_results, axis=0)
    num_components = np.argmin(mean_loss) + 1
    _LOGGER.info('After final grid search, optimal number of PCs is %d.',
                num_components)

    return num_components, mcv_results


def cross_validate_num_components(
        matrix: ExpMatrix, num_component_values: np.ndarray,
        max_num_components: int = 100,
        alpha: float = 1/9, capture_eff: float = 0.05, num_splits: int = 5,
        use_double_precision: bool = False,
        seed: int = 0) -> np.ndarray:
    """Perform MCV on grid of num_component values.""" 

    # this is the array to hold the loss values for all splits and # PCs
    L = np.zeros((num_splits, len(num_component_values)), dtype=np.float64)

    # generate fixed seeds for generating the training/validation datasets
    np.random.seed(seed)
    max_seed = np.iinfo(np.uint32).max
    split_seeds = np.random.randint(0, max_seed+1, size=num_splits)

    _LOGGER.info(
        'Testing grid of %d num_component values...',
        len(num_component_values))

    for i in range(num_splits):
        _LOGGER.info(
            'Now processing split %d/%d...', i+1, num_splits)

        train_matrix, val_matrix = generate_split(
            matrix, capture_eff, alpha, seed=split_seeds[i])

        # covert data to the right float data type
        if use_double_precision:
            train_matrix = train_matrix.astype(np.float64, copy=False)
            val_matrix = val_matrix.astype(np.float64, copy=False)
        else:
            train_matrix = train_matrix.astype(np.float32, copy=False)
            val_matrix = val_matrix.astype(np.float32, copy=False)

        # make sure memory is freed up
        gc.collect()

        # generate a CompressedData object
        pca_model = PCAModel(max_num_components)
        pc_scores = pca_model.fit_transform(train_matrix)
        num_transcripts = train_matrix.sum(axis=0)
        compressed_data = CompressedData(
            pca_model, pc_scores, num_transcripts)

        L[i, :] = cross_validate_split(
            compressed_data, val_matrix, num_component_values)

    return L


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
        # calculate Poisson loss using MCV here
        restored_matrix = compressed_data.decompress(
            num_components=num_components, apply_scaling=False)

        # scale the restored matrix to match the validation matrix
        restored_matrix = util.scale(restored_matrix, val_transcript_count)

        loss_values[i] = calculate_poisson_loss(restored_matrix, val_matrix)
        gc.collect()

    return loss_values
