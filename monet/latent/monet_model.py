# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `MonetModel` class."""

import logging
import os
import pickle
import time
import gc
from math import ceil

from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix
from . import PCAModel
from . import CompressedData
from .util import generate_split
from .util import cross_validate_split
from .util import aggregate_neighbors

_LOGGER = logging.getLogger(__name__)


class MonetModel:
    """A latent space model for single-cell RNA-Seq data."""
    
    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(
            self,
            max_num_components: int = 100,
            mcv_alpha: float = 1/9, mcv_capture_eff: float = 0.05,
            mcv_num_splits: int = 5, mcv_step_size: int = 10,
            mcv_fine_step_size: int = 3,
            num_components: int = None,
            agg_target_transcript_count: int = 200000,
            agg_max_frac_neighbors: float = 0.01,
            agg_num_neighbors: int = None,
            use_double_precision: bool = False,
            num_aggregation_steps: int = 1,
            seed: int = 0) -> None:

        self.max_num_components = max_num_components
        self.num_components = num_components

        self.mcv_alpha = mcv_alpha
        self.mcv_capture_eff = mcv_capture_eff
        self.mcv_num_splits = mcv_num_splits
        self.mcv_step_size = mcv_step_size
        self.mcv_fine_step_size = mcv_fine_step_size

        self.agg_target_transcript_count = agg_target_transcript_count
        self.agg_max_frac_neighbors = agg_max_frac_neighbors
        self.agg_num_neighbors = agg_num_neighbors

        self.use_double_precision = use_double_precision
        self.num_aggregation_steps = num_aggregation_steps
        self.seed = seed

        # important "parameters" that are learned
        self.num_components_ = None
        self.agg_num_neighbors_ = None
        self.agg_pca_model_ = None

        # main result
        self.pca_model_ = None

        # other results that we store
        self.mcv_results_ = None
        self.execution_time_ = None

    @property
    def num_cells_(self):
        return self.raw_num_transcripts_.size

    @property
    def num_genes_(self):
        return self.pca_model_.num_genes_

    @property
    def is_fitted(self):
        return (self.pca_model_ is not None)

    def _require_fitted_model(self):
        if not self.is_fitted:
            raise NotFittedError(
                'You must fit the model first using `fit()`.')

    def save_pickle(self, fpath: str) -> None:
        """Save Monet model to pickle file."""
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved Monet model to pickle file "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load Monet model from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded Monet model from pickle file "%s".', fpath)
        return clf


    def fit(self, matrix: ExpMatrix) -> None:
        """Fit Monet model."""

        t0_total = time.time()

        if not np.issubdtype(matrix.values.dtype, np.integer):
            raise ValueError(
                'Matrix data type must be integer! '
                'Try `matrix = matrix.astype(np.uint32)` before calling '
                '`fit()`.')

        self.raw_num_transcripts_ = matrix.sum(axis=0)

        ### Phase I: Estimate dimensionality
        _LOGGER.info('Beginning of Phase I (Estimate dimensionality)...')
        t0 = time.time()
        self._determine_num_components(matrix)
        t1 = time.time()
        _LOGGER.info('Phase I (Estimating dimensionality) took %.1f s.', t1-t0)

        ### Phase II: Learn latent space
        _LOGGER.info('Beginning of Phase II (Latent space inference)...')
        t0 = time.time()
        self._learn_latent_space(matrix)
        t1 = time.time()
        _LOGGER.info('Phase II (Latent space inference) took %.1f s.', t1-t0)

        t1_total = time.time()
        self.execution_time_ = t1_total - t0_total
        _LOGGER.info('Fitting the Monet model took %.1f s (%.1f min).',
                     self.execution_time_, self.execution_time_/60.0)


    def transform(self, matrix: ExpMatrix,
                  include_var: bool = False) -> ExpMatrix:

        self._require_fitted_model()

        pc_scores = self.pca_model_.transform(matrix, include_var)
        return pc_scores


    def fit_transform(self, matrix: ExpMatrix,
                      include_var: bool = False) -> ExpMatrix:
        self.fit(matrix)
        pc_scores = self.transform(matrix, include_var)
        return pc_scores


    def _determine_num_components(self, matrix: ExpMatrix) -> None:
        """Use MCV to determine the optimum number of PCs.
        
        Implements a grid search strategy to tune the number of PCs. 
        """
        if self.num_components is not None:
            self.num_components_ = self.num_components
            _LOGGER.info('Will use %d PCs (value was pre-specified).',
                         self.num_components_)
            return
        
        _LOGGER.info('Using molecular cross-validation to determine '
                     'the number of PCs...')

        if not np.issubdtype(matrix.values.dtype, np.integer):
            raise ValueError('Matrix data type must be integer!')

        # this is the array to hold the loss values for all splits and # PCs
        mcv_results = np.zeros((self.mcv_num_splits, self.max_num_components),
                    dtype=np.float64)
        mcv_results[:, :] = np.inf

        ## first round: test coarse grid of num_component values
        _LOGGER.info('Testing coarse grid of num_component values...')

        num_component_values = np.arange(
            self.mcv_step_size, self.max_num_components+1, self.mcv_step_size)

        L = self._cross_validate_num_components(matrix, num_component_values)

        mcv_results[:, num_component_values-1] = L
        gc.collect()

        mean_loss = np.mean(mcv_results, axis=0)
        coarse_num_components = np.argmin(mean_loss) + 1
        _LOGGER.info('Coarse grid search yielded optimum of %d PCs...',
                     coarse_num_components)

        ## second round: zoom in on range that contains the optimal value
        _LOGGER.info('Testing fine grid of num_component values...')

        # make sure we're not testing values larger than max_num_components
        largest_num_components = coarse_num_components + self.mcv_step_size - \
                self.mcv_fine_step_size
        largest_num_components = \
            min(largest_num_components, self.max_num_components)

        num_component_values = np.r_[
            np.arange(coarse_num_components - self.mcv_step_size + \
                    self.mcv_fine_step_size, coarse_num_components,
                    self.mcv_fine_step_size),
            np.arange(largest_num_components, coarse_num_components,
                    -self.mcv_fine_step_size)[::-1]]

        L = self._cross_validate_num_components(matrix, num_component_values)
        
        mcv_results[:, num_component_values-1] = L
        del L
        gc.collect()

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
        
        for i in range(fine_num_components, self.max_num_components):
            if np.isfinite(mean_loss[i]):
                break
            num_component_values.append(i+1)

        num_component_values = np.array(num_component_values, dtype=np.uint64)

        L = self._cross_validate_num_components(matrix, num_component_values)
        mcv_results[:, num_component_values-1] = L
        del L
        gc.collect()

        # determine final value for the optimal number of PCs
        mean_loss = np.mean(mcv_results, axis=0)
        num_components = np.argmin(mean_loss) + 1
        _LOGGER.info('After final grid search, optimal number of PCs is %d.',
                    num_components)
        
        self.num_components_ = num_components
        self.mcv_results_ = mcv_results


    def _cross_validate_num_components(
            self, matrix: ExpMatrix,
            num_component_values: np.ndarray) -> np.ndarray:
        """Perform MCV on grid of num_component values.""" 

        # this is the array to hold the loss values for all splits and # PCs
        L = np.zeros((self.mcv_num_splits, len(num_component_values)),
                     dtype=np.float64)

        # generate fixed seeds for generating the training/validation datasets
        np.random.seed(self.seed)
        max_seed = np.iinfo(np.uint32).max
        split_seeds = np.random.randint(
            0, max_seed+1, size=self.mcv_num_splits)

        _LOGGER.info(
            'Testing grid of %d num_component values...',
            len(num_component_values))

        for i in range(self.mcv_num_splits):
            _LOGGER.info(
                'Now processing split %d/%d...', i+1, self.mcv_num_splits)

            train_matrix, val_matrix = generate_split(
                matrix, self.mcv_capture_eff, self.mcv_alpha,
                seed=split_seeds[i])
            
            # covert data to the right float data type
            if self.use_double_precision:
                train_matrix = train_matrix.astype(np.float64, copy=False)
                val_matrix = val_matrix.astype(np.float64, copy=False)
            else:
                train_matrix = train_matrix.astype(np.float32, copy=False)
                val_matrix = val_matrix.astype(np.float32, copy=False)

            # make sure memory is freed up
            gc.collect()

            # generate a CompressedData object
            pca_model = PCAModel(self.max_num_components)
            pc_scores = pca_model.fit_transform(train_matrix)
            num_transcripts = train_matrix.sum(axis=0)
            compressed_data = CompressedData(
                pca_model, pc_scores, num_transcripts)

            L[i, :] = cross_validate_split(
                compressed_data, val_matrix, num_component_values)
        
        return L


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
                    'Will use num_neighbors=%d for aggregation'
                    '(value was determined automatically '
                    'based on a target transcript count of %d).',
                    num_neighbors, self.agg_target_transcript_count)
            else:
                _LOGGER.warning(
                    'Will use num_neighbors=%d for aggregation, '
                    'to not exceed %.1f %% of the total number of cells. '
                    'However, based on a target transcript count of %d, '
                    'we should use k=%d. As a result, gene loadings'
                    'may be biased towards highly expressed genes.',
                    max_num_neighbors, 100*self.agg_max_frac_neighbors,
                    self.agg_target_transcript_count, num_neighbors)
                num_neighbors = max_num_neighbors

        self.agg_num_neighbors_ = num_neighbors


    def _learn_latent_space(self, matrix: ExpMatrix) -> None:
        """Learn the latent space."""

        _LOGGER.info('Learning the latent space...')
        t0 = time.time()

        if self.num_components_ is None:
            raise RuntimeError(
                'Must call `_determine_num_components()` before '
                '`_learn_agg_latent_space().`')

        # Train initial PCA model
        if self.use_double_precision:
            float_matrix = matrix.astype(np.float64, copy=False)
        else:
            float_matrix = matrix.astype(np.float32, copy=False)

        transcript_count = np.median(float_matrix.sum(axis=0))
        latent_space = PCAModel(
            self.num_components_, transcript_count, seed=self.seed)
        pc_scores = latent_space.fit_transform(float_matrix)
        del float_matrix
        gc.collect()

        # Check if we perform aggregation steps
        if self.num_aggregation_steps == 0:
            _LOGGER.warning('Not performing any aggregation steps...')

            t1 = time.time()
            _LOGGER.info('Learned a %d-dimensional latent space in %.1f s.',
                         self.num_components_, t1-t0)

            self.pca_model_ = latent_space
            return

        # Determine number of neighbors for aggregation step
        self._determine_agg_num_neighbors(matrix)

        if self.agg_num_neighbors_ == 1:
            # Skip aggregation steps
            _LOGGER.warning('Skipping aggregation steps, because '
                            'agg_num_neighbors=1.')

        else:
            # Perform specified number of aggregation-PCA cycles
            for agg_step in range(self.num_aggregation_steps):

                _LOGGER.info('Now performing aggregation step %d/%d...',
                            agg_step+1, self.num_aggregation_steps)

                # kNN aggregation
                agg_matrix = aggregate_neighbors(
                    matrix, pc_scores, self.agg_num_neighbors_)

                if self.use_double_precision:
                    agg_matrix = agg_matrix.astype(np.float64, copy=False)
                else:
                    agg_matrix = agg_matrix.astype(np.float32, copy=False)
                gc.collect()
                
                latent_space = PCAModel(
                    self.num_components_, transcript_count, seed=self.seed)
                pc_scores = latent_space.fit_transform(agg_matrix)

                del agg_matrix
                gc.collect()

        t1 = time.time()
        _LOGGER.info('Learned a %d-dimensional latent space in %.1f s.',
                    self.num_components_, t1-t0)

        self.pca_model_ = latent_space


    def plot_mcv_results(self, show_optimum=True) -> go.Figure:

        self._require_fitted_model()

        mean = self.mcv_results_.mean(axis=0)
        #print(mean)

        x = np.arange(mean.size) + 1
        #x = x[9:30]
        #mean = mean[9:30]

        sel = np.isfinite(mean)

        x = x[sel]
        y = mean[sel]

        trace = go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=3, color='rgba(0.0,0.0,0.7,0.5)'),
            marker=dict(color='navy'),
        )

        data = [trace]
        
        ptp_y = np.ptp(y)
        
        #tickvals = np.arange(10, 101, 10)
        #tickvals = np.r_[[10], np.arange(20, 101, 20)]
        tickvals = np.r_[[1], np.arange(20, 101, 20)]
        
        min_loss = np.amin(y)
        min_index = np.argmin(y)
        opt_num_components = x[min_index]
        
        print(opt_num_components)
        
        annotations = []

        if show_optimum:
            annotations.append(
                dict(
                    xref='x',
                    yref='y',
                    x=opt_num_components,
                    y=min_loss+ptp_y*0.02,
                    text='<b>%d</b>' % opt_num_components,
                    align='center',
                    ax=0,
                    ay=-50,
                    showarrow=True,
                    font=dict(family='serif', size=32, color='black'),
                    #angle=90,
                    arrowcolor='gray',
                    arrowwidth=8,
                    arrowsize=0.4,
                    arrowhead=4))
        
        layout = go.Layout(
            width=800,
            height=600,
            font=dict(family='serif', size=32),
            xaxis=dict(automargin=False, linecolor='black', ticklen=5, ticks='outside', title='Number of PCs', showline=True, zeroline=False, tickvals=tickvals),
            yaxis=dict(automargin=False, linecolor='black', ticklen=5, showticklabels=False, title='Poisson loss', showline=True, zeroline=False),
            margin=dict(b=100, l=50),
            plot_bgcolor='white',
            annotations=annotations,
        )

        fig = go.Figure(data=data, layout=layout)
        return fig
