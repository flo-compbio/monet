# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `CompressedData` class."""

import logging
from typing import Union, Iterable
import time
import os
import pickle

from ..core import ExpMatrix
from ..latent import PCAModel
from .. import util

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.stats import poisson

_LOGGER = logging.getLogger(__name__)


class CompressedData:
    """Compressed single-cell RNA-Seq data."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(self, pca_model: PCAModel, pc_scores: pd.DataFrame,
                 num_transcripts: pd.Series) -> None:

        self.pca_model = pca_model
        self.pc_scores = pc_scores
        self.num_transcripts = num_transcripts

    @property
    def shape(self):
        return (self.num_genes, self.num_cells)

    @property
    def num_genes(self):
        return self.pca_model.num_genes_
    
    @property
    def num_cells(self):
        return self.pc_scores.shape[0]

    @property
    def num_components(self):
        return self.pca_model.num_components

    @property
    def matrix(self):
        return self.decompress()


    @classmethod
    def from_matrix(cls, matrix: ExpMatrix, num_components: int):
        """Generate a compressed matrix."""

        pca_model = PCAModel(num_components)
        pc_scores = pca_model.fit_transform(matrix)

        num_transcripts = matrix.sum(axis=0)

        new_size = pc_scores.shape[0] * pc_scores.shape[1] + \
                pca_model.num_genes_ * pca_model.num_components

        naive_size = matrix.shape[0] * matrix.shape[1]
        compression_factor = naive_size / new_size
        _LOGGER.info('Compressed data by a factor of %.1f.',
                     compression_factor)

        return cls(pca_model, pc_scores, num_transcripts)

    def save_pickle(self, fpath: str) -> None:
        """Save compressed data to pickle file."""
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved compressed data to pickle file "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load compressed data from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded compressed data from pickle file "%s".', fpath)
        return clf

    def decompress(
            self, num_components: int = None,
            apply_scaling: bool = True) -> ExpMatrix:
        """Extract the expression matrix."""

        # Reverse the PCA
        pc_scores = self.pc_scores
        if num_components is not None:
            if num_components > self.pc_scores.shape[1]:
                raise ValueError(
                    '"num_components" is greater than the number of PCs '
                    'available')

        matrix = self.pca_model.inverse_transform(pc_scores, num_components)

        if apply_scaling:
            # scale cells to right transcript counts
            matrix *= (self.num_transcripts / matrix.sum(axis=0))

        return matrix
