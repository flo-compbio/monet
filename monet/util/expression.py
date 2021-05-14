# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Utility functions for working with expression data."""

import hashlib

import pandas as pd
import numpy as np

from ..core import ExpMatrix


def scale(matrix: ExpMatrix, transcript_count=None) -> ExpMatrix:
    """Scale the cell expression profiles to constant transcript count.

    If `transcript_count` is is not provided, uses the median
    transcript count of all cells in the matrix."""
    num_transcripts = matrix.sum(axis=0)

    if transcript_count is None:
        transcript_count = num_transcripts.median()

    scaled_matrix = (transcript_count / num_transcripts) * matrix
    return scaled_matrix


def scale_genes(matrix: ExpMatrix, inplace: bool = False):
    
    gene_max = np.amax(matrix.values, axis=1)
    num_cells = matrix.shape[1]
    X = matrix.values.T / np.tile(gene_max, (num_cells, 1))
    
    if inplace:
        matrix.values[:, :] = X.T
    else:
        matrix = ExpMatrix(genes=matrix.index, cells=matrix.columns,
                           data=X.T)
    return matrix


def normalize_genes(matrix: ExpMatrix, inplace: bool = False):
    
    gene_ptp = np.ptp(matrix.values, axis=1)
    gene_min = np.min(matrix.values, axis=1)
    num_cells = matrix.shape[1]
    X = matrix.values.T - np.tile(gene_min, (num_cells, 1))
    X = X / np.tile(gene_ptp, (num_cells, 1))
    
    if inplace:
        matrix.values[:, :] = X.T
    else:
        matrix = ExpMatrix(genes=matrix.index, cells=matrix.columns,
                           data=X.T)
    return matrix


def center_genes(
        matrix: ExpMatrix,
        use_median: bool = False, inplace: bool = False) -> ExpMatrix:
    """Center gene expression values."""

    if use_median:
        mean = np.median(matrix.values, axis=1)
    else:
        mean = np.mean(matrix.values, axis=1)

    num_cells = matrix.shape[1]
    X = matrix.values.T - np.tile(mean, (num_cells, 1))
    
    if inplace:
        matrix.values[:, :] = X.T
        center_matrix = matrix
    else:
        center_matrix = ExpMatrix(
            genes=matrix.index, cells=matrix.columns, data=X.T)
    
    return center_matrix


def standardize_genes(matrix: ExpMatrix, inplace: bool = False) -> ExpMatrix:
    """Convert gene expression values to z-scores."""
    mean = np.mean(matrix.values, axis=1)
    std = np.std(matrix.values, axis=1, ddof=1)

    X = (matrix.values.T - mean) / std
    
    if inplace:
        matrix.values[:, :] = X.T
        zscore_matrix = matrix
    else:
        zscore_matrix = ExpMatrix(
            genes=matrix.index, cells=matrix.columns, data=X.T)

    return zscore_matrix


def calculate_hash(df: pd.DataFrame) -> str:
    """Calculates a unique hash for a pandas `DataFrame`."""
    index_str = ','.join(str(e) for e in df.index)
    col_str = ','.join(str(e) for e in df.columns)
    data_str = ';'.join([index_str, col_str]) + ';'
    data = data_str.encode('utf-8') + df.values.tobytes()
    return str(hashlib.md5(data).hexdigest())


def ft_transform(matrix: ExpMatrix) -> ExpMatrix:
    """Applies the Freeman-Tukey transformation to stabilize variance."""

    # work around a bug where np.sqrt() says input is invalid for arrays
    # of type np.float32 that contain zeros
    invalid_errstate = 'warn'
    if np.issubdtype(matrix.values.dtype, np.float32):
        if np.amin(matrix.values) >= 0:
            invalid_errstate = 'ignore'

    X = matrix.values
    with np.errstate(invalid=invalid_errstate):
        X = np.sqrt(X) + np.sqrt(X+1)

    tmatrix = ExpMatrix(genes=matrix.genes, cells=matrix.cells, data=X)

    return tmatrix


def pearson_transform(
        matrix: ExpMatrix, min_exp_thresh: float = 0.01) -> ExpMatrix:
    """Uses pearson residuals to stabilize variance."""

    invalid_errstate = 'warn'
    if np.issubdtype(matrix.values.dtype, np.float32):
        if np.amin(matrix.values) >= 0:
            invalid_errstate = 'ignore'

    scaled_matrix = matrix.scale()

    mean = np.mean(scaled_matrix.values, axis=1)
    mean[mean < min_exp_thresh] = min_exp_thresh

    with np.errstate(invalid=invalid_errstate):
        X = (scaled_matrix.values.T / np.sqrt(mean))

    tmatrix = ExpMatrix(genes=matrix.genes, cells=matrix.cells, data=X.T)
    return tmatrix
