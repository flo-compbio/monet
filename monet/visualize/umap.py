# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import time
import sys
from typing import Union, Tuple

import umap
import plotly.graph_objs as go
import pandas as pd
# import numpy as np

from ..core import ExpMatrix
from ..latent import PCAModel
from .cells import plot_cells, plot_cells_random_order

_LOGGER = logging.getLogger()

Numeric = Union[int, float]


def umap_plot(
        matrix: ExpMatrix, 
        num_components: int = 50, num_neighbors: int = 30,
        pca_model: PCAModel = None,
        min_dist: float = 0.3,
        transform_name: str = None,
        init: str = 'random', seed: int = 0,
        random_order: bool = False,
        umap_kwargs=None, **kwargs) \
            -> Tuple[go.Figure, pd.DataFrame]:
    """Perform UMAP on PCA-transformed data."""

    if umap_kwargs is None:
        umap_kwargs = {}

    if pca_model is None:
        _LOGGER.info(
            'No PCA model provided, performing PCA to determine first %d '
            'principal components...', num_components)
        pca_kwargs = {}
        if transform_name is not None:
            _LOGGER.info('Using "%s" transform!', transform_name)
            pca_kwargs['transform_name'] = transform_name
        pca_model = PCAModel(num_components=num_components, seed=seed,
                             **pca_kwargs)
        pc_scores = pca_model.fit_transform(matrix)

    else:
        _LOGGER.info(
            'Using PCA model to project data onto a %d-dimensional '
            'latent space...', pca_model.num_components_)
        pc_scores = pca_model.transform(matrix)


    # perform UMAP
    umap_seed = umap_kwargs.pop('random_state', seed)
    t0 = time.time()
    _LOGGER.info('Performing UMAP...'); sys.stdout.flush()
    model = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist,
                      random_state=umap_seed, init=init, **umap_kwargs)
    Y = model.fit_transform(pc_scores.values)
    t1 = time.time()
    _LOGGER.info('UMAP took %.1f s.', t1-t0)

    dim_labels = ['UMAP dim. %d' % (l+1)
                    for l in range(Y.shape[1])]

    umap_scores = pd.DataFrame(
        index=pc_scores.index, columns=dim_labels, data=Y)

    if random_order:
        fig, _ = plot_cells_random_order(umap_scores, **kwargs)

    else:
        fig = plot_cells(umap_scores, **kwargs)

    return fig, umap_scores
