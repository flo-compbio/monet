# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `SimENHANCE` class."""

from typing import Union
import os
import pickle
import logging

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.stats import hypergeom
from scipy.stats import bernoulli
from sklearn.exceptions import NotFittedError

from ..core import ExpMatrix
from ..latent import PCAModel
from ..denoise import ENHANCE

_LOGGER = logging.getLogger(__name__)


class SimENHANCE:
    """Simulation of single-cell RNA-Seq data."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(
            self, denoising_model: ENHANCE,
            max_eff_noise_factor: float = 2.0) -> None:

        self.denoising_model = denoising_model
        self.max_eff_noise_factor = max_eff_noise_factor

    @property
    def compressed_data(self):
        return self.denoising_model.compressed_data_

    @property
    def num_cells(self):
        return self.denoising_model.num_cells_

    @property
    def num_genes(self):
        return self.denoising_model.num_genes_

    @property
    def num_transcripts(self):
        return self.denoising_model.raw_num_transcripts_

    @property
    def efficiency_factors(self):
        return self.denoising_model.efficiency_factors_

    @property
    def cell_sizes(self):
        return self.denoising_model.cell_sizes_
    
    @property
    def true_matrix(self):
        return self.denoising_model.compressed_data_.decompress()


    def generate(self, seed: int = 0,
                 num_components: int = None) -> ExpMatrix:

        if not self.denoising_model.is_fitted:
            raise NotFittedError('Cannot simulate data because the denoising '
                                 'model has not been fitted yet.')

        denoised_matrix = self.denoising_model.compressed_data_.decompress(
            num_components=num_components)

        ## 1. Simulate efficiency noise

        # get efficiency factors and remove outliers (truncate values)
        eff_factors = self.denoising_model.efficiency_factors_.copy()
        max_factor = self.max_eff_noise_factor
        eff_factors.loc[eff_factors < 1 / max_factor] = 1 / max_factor
        eff_factors.loc[eff_factors > max_factor] = max_factor

        # use bootstrapping to simulate efficiency factors
        np.random.seed(seed)
        sel = np.random.choice(
            self.num_cells, size=self.num_cells, replace=True)
        f = eff_factors.values[sel].copy()
        
        ## 2. Simulate sampling noise

        # get simulated cell sizes
        sim_cell_sizes = f * self.cell_sizes.values
        sim_cell_sizes = sim_cell_sizes.round().astype(np.uint32)

        _LOGGER.info('Simulating data by sampling from the '
                        'Poisson distribution.')

        # scale true matrix to match the target transcript count
        true_matrix = denoised_matrix * \
                (sim_cell_sizes / denoised_matrix.values.sum(axis=0))
        
        # simulate data by sampling from Poisson distribution
        np.random.seed(seed)
        T = true_matrix.values.T
        X = np.empty(T.shape, dtype=np.uint32)
        for i in range(self.num_cells):
            X[i, :] = poisson.rvs(T[i, :])
        sim_matrix = ExpMatrix(
            X.T, genes=true_matrix.genes, cells=true_matrix.cells)

        _LOGGER.info('Generated simulated expression matrix with hash: %s',
                        sim_matrix.hash)

        return sim_matrix


    def save_pickle(self, fpath: str) -> None:
        """Save simulation to file in pickle format."""
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved simulation to "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load simulation from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded simulation from "%s".', fpath)
        return clf
