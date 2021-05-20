# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `PCAModel` class."""

import logging
import time
import os
import pickle
import gc
from typing import Union, Iterable

from ..core import ExpMatrix
from .. import util

from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


class PCAModel:
    """PCA model for single-cell RNA-Seq data."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(self, num_components=50,
                 transcript_count: Union[float, int] = None,
                 transform_name: str = 'freeman-tukey',
                 svd_solver = 'randomized',
                 sel_genes: Iterable[str] = None,
                 seed: int = 0) -> None:

        self.num_components = num_components
        self.transcript_count = transcript_count
        self.transform_name = transform_name
        self.svd_solver = svd_solver
        self.seed = seed
        self.sel_genes = sel_genes

        self.genes_ = None
        self.model_ = None
        self.dtype_ = None
        self.transcript_count_ = None
        self.component_labels_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None


    @property
    def num_genes_(self):
        return self.genes_.size

    @property
    def loadings_(self):
        # ensure column-major ordering
        L = np.array(self.model_.components_.T, 
                     order='F', copy=False)
        loadings = pd.DataFrame(
            data=L,
            index=self.genes_,
            columns=self.component_labels_)
        return loadings

    @property
    def total_var_(self):
        return self.model_.explained_variance_[0] / \
            self.model_.explained_variance_ratio_[0]


    def _require_trained_model(self):
        if self.model_ is None:
            raise NotFittedError(
                'You must train the model first using `fit()`.')        


    def fit_transform(self, matrix: ExpMatrix,
                      include_var: bool = False) -> pd.DataFrame:
        """Train the model and return PC scores."""

        # make sure matrix has floating point data type
        if not np.issubdtype(matrix.values.dtype, np.floating):
            matrix = matrix.astype(np.float32, copy=False)
            _LOGGER.info('Converted matrix to float32 data type.')

        # determine transcript count
        if self.transcript_count is not None:
            self.transcript_count_ = self.transcript_count
        else:
            self.transcript_count_ = matrix.median_transcript_count

        self.dtype_ = matrix.values.dtype

        # scale and apply transform
        matrix = matrix.scale(self.transcript_count_).transform(
            self.transform_name)
        gc.collect()

        # select genes (if desired)
        if self.sel_genes is not None:
            matrix = matrix.loc[self.sel_genes]

        # perform PCA
        model = PCA(n_components=self.num_components,
                    svd_solver=self.svd_solver,
                    random_state=self.seed)

        # make sure input data is C-contiguous
        X = np.array(matrix.X, order='C', copy=False)

        assert X.dtype == self.dtype_
        t0 = time.time()
        Y = model.fit_transform(X)
        t1 = time.time()
        _LOGGER.info('The PCA took %.1f s.', t1-t0)

        # make sure Y is F-contiguous
        # by default, output of PCA is not contiguous
        Y = np.array(Y, dtype=self.dtype_, order='F', copy=False)

        # convert to DataFrame
        total_var = X.var(axis=0, ddof=1).sum()
        explained_var = Y.var(axis=0, ddof=1)
        explained_var_ratio = explained_var / total_var
        
        if include_var:
            dim_labels = [
                'PC %d (%.1f %%)' % (c+1, 100*v)
                for c, v in zip(
                    range(self.num_components), explained_var_ratio)]
        else:
            dim_labels = ['PC %d' % (c+1) for c in range(self.num_components)]
        
        pc_scores = pd.DataFrame(Y, index=matrix.cells, columns=dim_labels)

        self.genes_ = matrix.genes.copy()
        self.model_ = model
        self.component_labels_ = dim_labels
        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = explained_var_ratio

        _LOGGER.info(
            'The fraction of variance explained by the %d selected PCs is '
            '%.1f %%.', self.num_components,
            100 * explained_var_ratio.sum())

        return pc_scores


    def fit(self, matrix: ExpMatrix) -> None:
        """Train the model."""

        self.fit_transform(matrix)


    def transform(self, matrix: ExpMatrix,
                  include_var: bool = False) -> pd.DataFrame:
        """Apply the model and return the PC scores."""

        self._require_trained_model()

        # make sure data is cast to same data type as training data
        matrix = matrix.astype(self.dtype_, copy=False)

        # apply scaling and transformation
        _LOGGER.info('Expression profiles will be scaled %.2fx (on average).',
                     self.transcript_count_ / matrix.median_transcript_count)
        matrix = matrix.scale(self.transcript_count_).transform(
            self.transform_name)
        gc.collect()

        # align genes
        num_missing = self.num_genes_ - self.genes_.isin(matrix.genes).sum()
        if num_missing > 0:
            frac_missing = num_missing / matrix.num_genes
            _LOGGER.warning('No expression data for %d / %d genes (%.1f %%) '
                            'in the PCA model.',
                            num_missing, matrix.num_genes, 100*frac_missing)
        matrix = matrix.reindex(index=self.genes_, fill_value=0)

        # project onto principal components
        X = matrix.X
        assert X.dtype == self.dtype_
        Y = self.model_.transform(X)

        # make sure score matrix is F-contiguous
        Y = np.array(Y, order='F', copy=False)
        pc_var = Y.var(axis=0, ddof=1)
        total_var = X.var(axis=0, ddof=1).sum()
        explained_variance_ratio = pc_var / total_var

        _LOGGER.info(
            'Projection onto %d PCs retained %.1f %% of the total '
            'variance in the scaled and FT-transformed data.',
            self.num_components, 100*(explained_variance_ratio.sum()))

        if include_var:
            dim_labels = ['PC %d (%.1f %%)'
                    % (c+1, 100*explained_variance_ratio[c])
                    for c in range(self.num_components)]
        else:
            dim_labels = ['PC %d' % (c+1) for c in range(self.num_components)]

        # generate DataFrame
        pc_scores = pd.DataFrame(Y, index=matrix.cells, columns=dim_labels)

        return pc_scores


    def inverse_transform(self, pc_scores: pd.DataFrame,
                          num_components: int = None) -> ExpMatrix:
        """Inverse the PCA and restores untransformed expression values."""

        self._require_trained_model()

        # make sure data is cast to same data type as training data
        pc_scores = pc_scores.astype(self.dtype_, copy=False)

        Y = pc_scores.values
        L = self.model_.components_

        if num_components is not None:
            Y = Y[:, :num_components]
            L = L[:num_components: ]

        if Y.shape[1] < L.shape[0]:
            _LOGGER.warning(
                'The score matrix contains fewer components than the model. '
                'Will only use the first %d components of the model.',
                Y.shape[1])
            L = L[:Y.shape[1], :]

        # inverse PCA transform
        X = Y.dot(L)

        # make sure result is C-contiguous
        X = np.array(X, order='C', copy=False)

        # add gene means
        X = X + self.model_.mean_

        if self.transform_name == 'freeman-tukey':
            # reverse FT transform, y=sqrt(x)+sqrt(x+1)

            # values below 1 result in undefined/non-sensical values
            X[X < 1] = 1  

            # apply inverse
            X = np.power(X, 2.0)
            X = np.power(X - 1.0, 2.0) / (4.0 * X)

        elif self.transform_name == 'anscombe':
            # reverse anscombe transform, y=2*sqrt(x+3/8)

            # values below sqrt(1.5) result in negative/non-sensical values
            low_thresh = pow(1.5, 0.5)
            X[X < low_thresh] = low_thresh

            # apply inverse
            X = 0.125 * (2.0*np.power(X, 2.0) - 3.0)

        elif self.transform_name == 'log':
            # reverse log transform, y=ln(x+1)

            # values below 0 result in negative values
            X[X < 0] = 0

            # apply inverse
            X = np.exp(X) - 1

        # convert to ExpMatrix
        matrix = ExpMatrix(X.T, genes=self.genes_, cells=pc_scores.index)

        return matrix


    def save_pickle(self, fpath: str) -> None:
        """Save PCA model to file in pickle format."""
        #pred.write_pickle('')
        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved PCA model to "%s".', fpath)


    @classmethod
    def load_pickle(cls, fpath: str):
        """Load PCA model from pickle file."""
        with open(os.path.expanduser(fpath), 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded PCA model from "%s".', fpath)
        return clf
