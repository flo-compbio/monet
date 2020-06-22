# Author: Florian Wagner <florian.wagner@uchicago.edu>
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


def zscore(matrix: ExpMatrix) -> ExpMatrix:
    """Converts the matrix to z-scores."""
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1)
    zscore_matrix = ((matrix.T - mean)/std).T
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


#def combine_matrices(**kwargs):
#   for name, matrix in kwargs.items():
            