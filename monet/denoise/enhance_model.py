# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `ENHANCE` class."""

from typing import Tuple, Union
import time
import logging
import gc
import os
import pickle

from sklearn.exceptions import NotFittedError
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..latent import PCAModel
from ..latent import CompressedData
from .mcv import mcv_estimate_num_components
from .util import var_estimate_num_components
from .util import determine_num_neighbors
from .util import aggregate_neighbors

_LOGGER = logging.getLogger(__name__)


class EnhanceModel:
    """ENHANCE denoising model for single-cell RNA-Seq data."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(
            self,
            num_components: Union[int, str] = 'mcv',
            pca_model = None,
            pca_max_num_components: int = 100,
            pca_var_fold_thresh: float = 2.0,
            agg_target_transcript_count: Union[int, float] = 200000,
            agg_max_frac_neighbors: float = 0.01,
            agg_num_neighbors: int = None,
            skip_aggregation_step: bool = False,
            use_double_precision = False,
            seed: int = 0) -> None:

        self.pca_model = pca_model

        self.pca_max_num_components = pca_max_num_components
        self.pca_var_fold_thresh = pca_var_fold_thresh
        self.num_components = num_components

        self.skip_aggregation_step = skip_aggregation_step

        self.agg_target_transcript_count = agg_target_transcript_count
        self.agg_max_frac_neighbors = agg_max_frac_neighbors
        self.agg_num_neighbors = agg_num_neighbors

        self.use_double_precision = use_double_precision
        self.seed = seed

        # important "parameters" that are learned
        self.num_components_ = None
        self.pca_model_ = None
        self.agg_num_neighbors_ = None

        # main result
        self.compressed_data_ = None

        # other results that we store
        self.mcv_results_ = None
        self.raw_num_transcripts_ = None
        self.execution_time_ = None


    @property
    def num_cells_(self):
        return self.compressed_data_.num_cells


    @property
    def num_genes_(self):
        return self.compressed_data_.num_genes


    @property
    def cell_sizes_(self):
        return self.compressed_data_.num_transcripts


    @property
    def efficiency_factors_(self):
        return self.raw_num_transcripts_ / self.cell_sizes_


    @property
    def denoised_matrix_(self):
        denoised_matrix = self.compressed_data_.decompress()
        return denoised_matrix


    @property
    def denoised_agg_matrix_(self):
        denoised_agg_matrix = self.compressed_data_.decompress(
            apply_scaling=False)
        return denoised_agg_matrix


    @property
    def is_fitted(self):
        return self.compressed_data_ is not None


    @property
    def agg_pca_model_(self):
        return self.compressed_data_.pca_model


    def _require_fitted_model(self):
        if not self.is_fitted:
            raise NotFittedError(
                'You must train the model first using `fit()`.')


    def save_pickle(self, fpath: str) -> None:
        """Save denoising model to pickle file."""
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info(
            'Saved ENHANCE denoising model to pickle file "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load denoising model from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info(
            'Loaded ENHANCE denoising model from pickle file "%s".', fpath)
        return clf


    def save(self, *args, **kwargs) -> None:
        self.save_pickle(*args, **kwargs)


    @classmethod
    def load(cls, *args, **kwargs):
        return cls.load_pickle(*args, **kwargs)


    def fit_transform(self, matrix: ExpMatrix) -> ExpMatrix:
        """Perform denoising."""

        t0_total = time.time()

        if not np.issubdtype(matrix.values.dtype, np.integer):
            raise ValueError(
                'Matrix data type must be integer! '
                'Try `matrix = matrix.astype(np.uint32)` before calling '
                'fit().')
        gc.collect()

        raw_num_transcripts = matrix.sum(axis=0)

        ### Phase I: Determine number of PCs
        if isinstance(self.num_components, (int, np.integer)):
            _LOGGER.info('Skipping Phase I (Determine dimensionality), '
                         'because the number of PCs was specified.')
            num_components = self.num_components
            mcv_results = None

        else:
            _LOGGER.info('Beginning of Phase I (Determine dimensionality)...')
            t0 = time.time()
            num_components, mcv_results = \
                    self._determine_num_components(matrix)
            t1 = time.time()
            _LOGGER.info('Completed Phase I (Determine dimensionality) in '
                         '%.1f s.', t1-t0)

        ### Phase II: Aggregation step
        if self.skip_aggregation_step:
            _LOGGER.info('Skipping Phase II (Aggregation step), as specified.')
            pca_model = None
            agg_num_neighbors = None
            agg_matrix = matrix
            cell_sizes = raw_num_transcripts

        else:
            _LOGGER.info('Beginning of Phase II (Aggregation step)...')
            pca_model, agg_num_neighbors, agg_matrix, cell_sizes = \
                    self._aggregation_step(matrix, num_components)
            _LOGGER.info('Completed Phase II (Aggregation step).')

        ### Phase III: PCA step
        _LOGGER.info('Beginning of Phase III (PCA step)...')
        t0 = time.time()
        agg_pca_model = PCAModel(num_components, seed=self.seed)
        agg_pc_scores = agg_pca_model.fit_transform(agg_matrix)

        # extract principal components
        compressed_data = CompressedData(
            agg_pca_model, agg_pc_scores, cell_sizes)
        denoised_matrix = compressed_data.decompress()

        t1 = time.time()
        _LOGGER.info('Completed Phase III (PCA step) in %.1f s.', t1-t0)

        del agg_matrix
        gc.collect()

        ### Store results
        self.raw_num_transcripts_ = raw_num_transcripts

        self.num_components_ = num_components
        self.mcv_results_ = mcv_results
        self.pca_model_ = pca_model

        self.agg_num_neighbors_ = agg_num_neighbors
        self.compressed_data_ = compressed_data

        t1_total = time.time()
        self.execution_time_ = t1_total - t0_total
        _LOGGER.info('Denoising with ENHANCE took %.1f s (%.1f min).',
                     self.execution_time_, self.execution_time_/60.0)

        return denoised_matrix


    def fit(self, matrix: ExpMatrix) -> None:
        self.fit_transform(matrix)


    def _determine_num_components(self, matrix: ExpMatrix) \
            -> Tuple[int, np.ndarray]:

        if self.num_components is None or \
                self.num_components.lower() == 'mcv':
            _LOGGER.info('The number of PCs will be determined using '
                            'molecular cross-validation.')

            num_components, mcv_results = mcv_estimate_num_components(
                matrix, self.pca_max_num_components,
                use_double_precision=self.use_double_precision,
                seed=self.seed)

        elif self.num_components.lower() == 'variance-threshold':
            # use variance threshold based on noise matrix
            _LOGGER.info('The number of PCs will be determined using '
                         'a variance threshold.')
            num_components, _ = var_estimate_num_components(
                matrix, self.pca_max_num_components,
                self.pca_var_fold_thresh, self.seed)
            mcv_results = None

        else:
            raise ValueError(
                'Invalid value for `pca_num_components` (%s). '
                'Valid values are None, an integer, "variance-threshold", '
                'or "mcv".' % str(self.num_components))

        return num_components, mcv_results


    def _aggregation_step(self, matrix: ExpMatrix, num_components: int) \
                -> ExpMatrix:
        """Perform the aggregation step."""

        if self.pca_model is None:

            _LOGGER.info('Training a %d-dimensional PCA model for '
                         'kNN-aggregation...', num_components)
            ### 2. Train PCA model
            if self.use_double_precision:
                float_matrix = matrix.astype(np.float64, copy=False)
            else:
                float_matrix = matrix.astype(np.float32, copy=False)

            transcript_count = np.median(float_matrix.sum(axis=0))
            pca_model = PCAModel(num_components, transcript_count,
                                 seed=self.seed)
            pca_model.fit(float_matrix)
            del float_matrix; gc.collect()

        else:
            _LOGGER.info('Will perform kNN-aggregation using pre-specified '
                         '%d-dimensional PCA model.',
                         self.pca_model.num_components)
            pca_model = self.pca_model

        ### 3. Determine number of nearest neighbors
        if self.agg_num_neighbors is not None:
            agg_num_neighbors = self.agg_num_neighbors
            _LOGGER.info(
                'Will perform kNN-aggregation with num_neighbors=%d '
                '(value was specified).', agg_num_neighbors)

        else:
            agg_num_neighbors = determine_num_neighbors(
                matrix, self.agg_target_transcript_count,
                self.agg_max_frac_neighbors)

        ### 4. Aggregate neighbors
        _LOGGER.info('Performing nearest-neighbor aggregation with '
                     '%d neighbors...', agg_num_neighbors)
        pc_scores = pca_model.transform(matrix)
        agg_matrix, cell_sizes, _, _ = \
                aggregate_neighbors(matrix, pc_scores, agg_num_neighbors)

        return pca_model, agg_num_neighbors, agg_matrix, cell_sizes


    def plot_mcv_results(self, show_optimum=True) -> go.Figure:

        self._require_fitted_model()

        mean = self.mcv_results_.mean(axis=0)
        x = np.arange(mean.size) + 1

        sel = np.isfinite(mean)

        x = x[sel]
        y = mean[sel]

        trace = go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=3, color='rgba(0.0,0.0,0.7,0.5)'),
            marker=dict(color='navy'))

        data = [trace]

        ptp_y = np.ptp(y)

        tickvals = np.r_[[1], np.arange(20, 101, 20)]

        min_loss = np.amin(y)
        min_index = np.argmin(y)
        opt_num_components = x[min_index]

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
            xaxis=dict(
                automargin=False, linecolor='black',
                ticklen=5, ticks='outside',
                title='Number of PCs',
                showline=True, zeroline=False,
                tickvals=tickvals),
            yaxis=dict(
                automargin=False, linecolor='black',
                ticklen=5, showticklabels=False,
                title='Poisson loss',
                showline=True, zeroline=False),
            margin=dict(b=100, l=50),
            plot_bgcolor='white',
            annotations=annotations)

        fig = go.Figure(data=data, layout=layout)
        return fig
