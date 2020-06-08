# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `ENHANCE` class."""

from typing import Tuple, Union
import time
import logging
import gc
import copy
import os
import pickle
from math import ceil

from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

from ..core import ExpMatrix
from ..latent import PCAModel
from ..latent import CompressedData

from .. import util
from ..latent import MonetModel
from .util import aggregate_neighbors
from .util import estimate_num_components

_LOGGER = logging.getLogger(__name__)


class ENHANCE:
    """ENHANCE denoising model for single-cell RNA-Seq data."""
    
    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(
            self,
            pca_max_num_components: int = 100,
            pca_var_fold_thresh: float = 2.0,
            pca_num_components: int = None,
            agg_target_transcript_count: Union[int, float] = 200000,
            agg_max_frac_neighbors: float = 0.01,
            agg_num_neighbors: int = None,
            skip_aggregation_step: bool = False,
            skip_pca_step: bool = False,
            use_double_precision = False,
            monet_kwargs: dict = None,
            seed: int = 0) -> None:

        if monet_kwargs is None:
            monet_kwargs = {}

        self.pca_max_num_components = pca_max_num_components
        self.pca_var_fold_thresh = pca_var_fold_thresh
        self.pca_num_components = pca_num_components

        self.skip_aggregation_step = skip_aggregation_step
        self.skip_pca_step = skip_pca_step

        self.agg_target_transcript_count = agg_target_transcript_count
        self.agg_max_frac_neighbors = agg_max_frac_neighbors
        self.agg_num_neighbors = agg_num_neighbors

        self.use_double_precision = use_double_precision
        self.monet_kwargs = monet_kwargs
        self.seed = seed

        # important "parameters" that are learned
        self.pca_num_components_ = None
        self.agg_num_neighbors_ = None

        self.monet_model_ = None

        # main result
        self.compressed_data_ = None

        # other results that we store
        self.agg_num_transcripts_ = None
        self.raw_num_transcripts_ = None
        self.agg_cell_sizes_ = None
        self.agg_neighbors_ = None
        self.agg_neighbor_dists_ = None 
        self.execution_time_ = None


    @property
    def num_cells_(self):
        return self.cell_sizes_.size

    @property
    def num_genes_(self):
        return self.agg_pca_model_.num_genes_

    @property
    def efficiency_factors_(self):
        return self.raw_num_transcripts_ / self.cell_sizes_

    @property
    def denoised_matrix_(self):
        denoised_matrix = self.compressed_data_.decompress()
        return denoised_matrix

    @property
    def is_fitted(self):
        return (self.monet_model_ is not None)


    def _require_fitted_model(self):
        if not self.is_fitted:
            raise NotFittedError(
                'You must train the model first using `fit()`.')


    def save_pickle(self, fpath: str) -> None:
        """Save denoising model to pickle file."""
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved denoising model to pickle file "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load denoising model from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded denoising model from pickle file "%s".', fpath)
        return clf


    def fit_transform(self, matrix: ExpMatrix) -> ExpMatrix:
        """Perform denoising."""

        t0_total = time.time()

        if not np.issubdtype(matrix.values.dtype, np.integer):
            raise ValueError(
                'Matrix data type must be integer! '
                'Try `matrix = matrix.astype(np.uint32)` before calling fit().')
        gc.collect()

        self.raw_num_transcripts_ = matrix.sum(axis=0)

        ### Phase I: Setting things up
        _LOGGER.info('Beginning of Phase I (Dimensionality estimation)...')

        ## Determine number of PCs
        self._determine_num_components(matrix)
        gc.collect()

        _LOGGER.info('Completed Phase I (Dimensionality estimation).')

        ### Phase II: Aggregation step
        if self.skip_aggregation_step:
            agg_matrix = matrix
            self.cell_sizes_ = self.raw_num_transcripts_
            _LOGGER.info('Phase II (Aggregation step) has been skipped!')

        else:
            _LOGGER.info('Beginning of Phase II (Aggregation step)...')

            # determine number of neighbors for aggregation step
            self._determine_agg_num_neighbors(matrix)

            # aggregate neighbors
            agg_matrix = self._aggregation_step(matrix)

            _LOGGER.info('Completed Phase II (Aggregation step).')

        ### Phase III: PCA step
        if self.skip_pca_step:
            num_transcripts = agg_matrix.sum(axis=0)
            denoised_matrix = (self.cell_sizes_ / num_transcripts) * agg_matrix
            _LOGGER.info('Phase III (PCA step) has been skipped!')
        
        else:
            _LOGGER.info('Beginning of Phase III (PCA step)...')
            t0 = time.time()
            pca_model = PCAModel(self.pca_num_components_, seed=self.seed)
            pc_scores = pca_model.fit_transform(agg_matrix)

            # extract principal components
            compressed_data = CompressedData(
                pca_model, pc_scores, self.cell_sizes_)
            denoised_matrix = compressed_data.decompress()

            self.compressed_data_ = compressed_data
            t1 = time.time()
            _LOGGER.info('Completed Phase III (PCA step) in %.1f s.', t1-t0)

        del agg_matrix
        gc.collect()

        t1_total = time.time()
        self.execution_time_ = t1_total - t0_total
        _LOGGER.info('Denoising with ENHANCE took %.1f s (%.1f min).',
                     self.execution_time_, self.execution_time_/60.0)

        return denoised_matrix


    def fit(self, matrix: ExpMatrix) -> None:
        self.fit_transform(matrix)


    def _determine_num_components(self, matrix: ExpMatrix) -> None:

        if self.pca_num_components is not None:
            _LOGGER.info('Will use %d principal components '
                         '(value was pre-specified).', self.pca_num_components)
            self.pca_num_components_ = self.pca_num_components
        else:
            num_components, _ = estimate_num_components(
                matrix, self.pca_max_num_components, self.pca_var_fold_thresh,
                self.seed)
            self.pca_num_components_ = num_components


    def _determine_agg_num_neighbors(self, matrix: ExpMatrix) -> None:
        """Determine the number of neighbors to use for kNN-aggregation."""

        transcript_count = matrix.median_transcript_count
        _LOGGER.info('The median transcript count is %.1f.',
                     transcript_count)

        if self.agg_num_neighbors is not None:
            num_neighbors = self.agg_num_neighbors
            _LOGGER.info('Will perform kNN-aggregation with num_neighbors=%d '
                         '(value was pre-specified).', num_neighbors)

        else:
            num_neighbors = int(ceil(self.agg_target_transcript_count / 
                                     transcript_count))
            max_num_neighbors = \
                    max(int(self.agg_max_frac_neighbors * matrix.shape[1]), 1)
            if num_neighbors <= max_num_neighbors:
                _LOGGER.info(
                    'Will perform denoising with k=%d '
                    '(value was determined automatically '
                    'based on a target transcript count of %d).',
                    num_neighbors, self.agg_target_transcript_count)
            else:
                _LOGGER.warning(
                    'Performing denoising with k=%d, to not exceed %.1f %% of '
                    'the total number of cells. However, based on a target '
                    'transcript count of %d, we should use k=%d. As a result, '
                    'denoising results may be biased towards highly expressed '
                    'genes.',
                    max_num_neighbors, 100*self.agg_max_frac_neighbors,
                    self.agg_target_transcript_count, num_neighbors)
                num_neighbors = max_num_neighbors

        self.agg_num_neighbors_ = num_neighbors


    def _learn_agg_pca_model(self, matrix: ExpMatrix) -> None:
        """Learn the PCA model for kNN-aggregation."""

        _LOGGER.info('Learning a latent space for determining neighbors...')

        if self.pca_num_components_ is None:
            raise RuntimeError(
                'Must call `_determine_num_components()` before '
                '`_learn_agg_latent_space().`')

        if self.agg_num_neighbors_ is None:
            raise RuntimeError(
                'Must call `_determine_agg_neighbors()` before '
                '`_learn_agg_latent_space().`')

        _LOGGER.info('Using `num_components=%d` and `num_neighbors=%d`.',
                     self.pca_num_components_, self.agg_num_neighbors_)

        t0 = time.time()

        ## train initial PCA model
        if not np.issubdtype(matrix.values.dtype, np.floating):
            if self.use_double_precision:
                float_matrix = matrix.astype(np.float64, copy=False)
            else:
                float_matrix = matrix.astype(np.float32, copy=False)

        transcript_count = np.median(matrix.sum(axis=0))
        initial_latent_space = PCAModel(
            self.pca_num_components_, transcript_count, seed=self.seed)

        pc_scores = initial_latent_space.fit_transform(float_matrix)

        if self.agg_num_neighbors_ > 1:
            # train final PCA model on aggregated matrix
            agg_matrix, _, _, _ = aggregate_neighbors(
                matrix, pc_scores, self.agg_num_neighbors_)

            if self.use_double_precision:
                agg_matrix = agg_matrix.astype(np.float64, copy=False)
            else:
                agg_matrix = agg_matrix.astype(np.float32, copy=False)

            latent_space = PCAModel(
                self.pca_num_components_, transcript_count, seed=self.seed)
            latent_space.fit(agg_matrix)
            del agg_matrix
            gc.collect()

        else:
            # do not perform aggregation, initial model is final model
            _LOGGER.warning('Skipping aggregation step (k=1)!')
            latent_space = initial_latent_space

        t1 = time.time()
        _LOGGER.info('Learned a %d-dimensional latent space in %.1f s.',
                     self.pca_num_components_, t1-t0)

        self.agg_pca_model_ = latent_space


    def _aggregation_step(self, matrix: ExpMatrix) -> ExpMatrix:
        """Perform the aggregation step."""

        if self.agg_num_neighbors_ is None:
            raise RuntimeError(
                'Must call `_determine_agg_neighbors()` before '
                '`_aggregation_step().`')

        #if self.agg_pca_model_ is None:
        #    raise RuntimeError(
        #        'Must call `_learn_agg_pca_model()` before '
        #        '`_aggregation_step().`')

        _LOGGER.info('Inferring a %d-dimensional latent space...',
                     self.pca_num_components_)

        # collect parameters for Monet model
        monet_kwargs = copy.deepcopy(self.monet_kwargs)

        # make sure Monet model uses same number of PCs as the ENHANCE model
        monet_kwargs['num_components'] = self.pca_num_components_

        # make sure Monet model uses the same number of neareast neighbors
        # as ENHANCE model
        monet_kwargs['agg_num_neighbors'] = self.agg_num_neighbors_

        # make sure Monet model uses the same data type as ENHANCE model
        monet_kwargs['use_double_precision'] = self.use_double_precision

        # make sure Monet model uses the same seed as ENHANCE model
        monet_kwargs['seed'] = self.seed

        # intialize and fit Monet model
        monet_model = MonetModel(**monet_kwargs)
        pc_scores = monet_model.fit_transform(matrix)
        self.monet_model_ = monet_model

        _LOGGER.info('Performing nearest-neighbor aggregation with '
                     '%d neighbors...', self.agg_num_neighbors_)

        # aggregate neighbors
        agg_matrix, cell_sizes, neighbors, neighbor_dists = \
                aggregate_neighbors(matrix, pc_scores, self.agg_num_neighbors_)

        # store results, except for agg_matrix
        self.agg_num_transcripts = agg_matrix.sum(axis=0)
        self.cell_sizes_ = cell_sizes
        self.neighbors_ = neighbors
        self.neighbor_dists_ = neighbor_dists 

        return agg_matrix
