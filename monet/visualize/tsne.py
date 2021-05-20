# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import time
import sys
from typing import Union, Tuple, Dict, Iterable

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from ..core import ExpMatrix
from ..latent import PCAModel
from ..batch_correct import correct_mnn
from .cells import plot_cells, plot_cells_random_order
from .util import ACCESSIBLE_COLORS

_LOGGER = logging.getLogger()

Numeric = Union[int, float]


def tsne_plot(
        matrix: ExpMatrix, num_components: int = 50, 
        perplexity: Numeric = 30,
        pca_model: PCAModel = None,
        transform_name: str = None, sel_genes: Iterable[str] = None,
        init: str = 'random', seed: int = 0,
        exaggerated_tsne: bool = False, random_order: bool = False,
        learning_rate: float = 200.0,
        tsne_kwargs=None, **kwargs) \
            -> Tuple[go.Figure, pd.DataFrame]:
    """Perform t-SNE on PCA-transformed data."""

    if tsne_kwargs is None:
        tsne_kwargs = {}

    if pca_model is None:
        _LOGGER.info(
            'No PCA model provided, performing PCA to determine first %d '
            'principal components...', num_components)
        pca_kwargs = {}
        if transform_name is not None:
            _LOGGER.info('Using "%s" transform!', transform_name)
            pca_kwargs['transform_name'] = transform_name
        pca_model = PCAModel(
            num_components=num_components, sel_genes=sel_genes,
            seed=seed, **pca_kwargs)
        pc_scores = pca_model.fit_transform(matrix)

    else:
        _LOGGER.info(
            'Using PCA model to project data onto a %d-dimensional '
            'latent space...', pca_model.num_components)
        pc_scores = pca_model.transform(matrix)

    if learning_rate is None:
        learning_rate = max(200.0, matrix.shape[1] / 48.0)
        _LOGGER.info('Using learning_rate=%.1f', learning_rate)

    if exaggerated_tsne:
        init = 'random'
        tsne_kwargs['early_exaggeration'] = 4
        tsne_kwargs['n_iter'] = 250

    tsne_seed = tsne_kwargs.pop('random_state', seed)
    tsne_model = TSNE(perplexity=perplexity, random_state=tsne_seed,
                      learning_rate=learning_rate, init=init, **tsne_kwargs)

    t0 = time.time()
    if exaggerated_tsne:
        _LOGGER.info('Performing exaggerated t-SNE...'); sys.stdout.flush()
    else:
        _LOGGER.info('Performing t-SNE...'); sys.stdout.flush()

    Y = tsne_model.fit_transform(pc_scores.values)
    t1 = time.time()
    _LOGGER.info('t-SNE took %.1f s.' % (t1-t0))
    
    if exaggerated_tsne:
        dim_labels = ['t-SNE* dim. %d' % (l+1)
                      for l in range(Y.shape[1])]
    else:
        dim_labels = ['t-SNE dim. %d' % (l+1)
                      for l in range(Y.shape[1])]

    tsne_scores = pd.DataFrame(
        index=pc_scores.index, columns=dim_labels, data=Y)

    if random_order:
        fig, _ = plot_cells_random_order(tsne_scores, **kwargs)
    
    else:
        fig = plot_cells(tsne_scores, **kwargs)
    
    return fig, tsne_scores


def batch_corrected_tsne_plot(
        pca_model: PCAModel,
        ref_matrix: ExpMatrix, target_matrix: ExpMatrix,
        k: int = 20, num_mnn: int = 5,
        perplexity: Numeric = 30, seed: int = 0,
        marker_size: Numeric = 2.5, exaggerated_tsne: bool = True,
        cluster_colors: Dict[str, str] = None,
        tsne_kwargs: Dict = None, **kwargs) \
            -> Tuple[go.Figure, go.Figure, pd.DataFrame, pd.DataFrame]:
    """Perform a t-SNE after batch correction using mutual nearest neighbors.

    This function implements a method similar to the one proposed
    by Haghverdi et al. (Nat Biotech, 2018). For more details, see the doc
    string of the `correct_mnn()` function in the `monet.batch.mutual` module.

    Returns:
    ========
    1. go.Figure
        The t-SNE plot with cells plotted in random order.

    2. go.Figure
        The t-SNE plot with target cells plotted on top of reference cells.
        The purpose of this plot is to obtain a figure legend.

    3. pd.DataFrame
        The t-SNE scores.

    4. pd.DataFrame
        The PC scores used to generate the t-SNE.

    Notes:
    ======
    By default, an exagerrated t-SNE is performed that resembles UMAP.

    The t-SNE scores and the PC scores contain data for the cells from both the
    reference and the target expression matrix. The reference cell indices
    start with "Reference_" and the target cell indices start with "Target_".

    The PC scores for the target cells are the batch-corrected scores. The PC
    scores for the reference cells are simply the scores obtained by applying
    the PCA model.
    """

    if tsne_kwargs is None:
        tsne_kwargs = {}

    if cluster_colors is None:
        cluster_colors = {}

    if 'Reference' not in cluster_colors:
        cluster_colors['Reference'] = ACCESSIBLE_COLORS[3]
    if 'Target' not in cluster_colors:
        cluster_colors['Target'] = ACCESSIBLE_COLORS[2]

    # get batch-corrected PC scores for target matrix
    # and regular PC scores for reference matrix
    target_pc_scores, ref_pc_scores = correct_mnn(
        pca_model, ref_matrix, target_matrix, k=k, num_mnn=num_mnn)

    # combine both data frames
    ref_pc_scores.index = ref_pc_scores.index.to_series().apply(
        lambda x: 'Reference_' + x).values
    target_pc_scores.index = target_pc_scores.index.to_series().apply(
        lambda x: 'Target_' + x).values
    pc_scores = pd.concat([ref_pc_scores, target_pc_scores], axis=0)

    tsne_seed = tsne_kwargs.pop('random_state', seed)
    tsne_init = tsne_kwargs.pop('init', 'random')
    tsne_perp = tsne_kwargs.pop('perplexity', perplexity)

    t0 = time.time()
    if exaggerated_tsne:
        _LOGGER.info('Performing exaggerated t-SNE...'); sys.stdout.flush()
        tsne_model = TSNE(
            perplexity=tsne_perp, random_state=tsne_seed,
            early_exaggeration=4, n_iter=250,
            init=tsne_init, **tsne_kwargs)
    
    else:
        _LOGGER.info('Performing t-SNE...'); sys.stdout.flush()
        tsne_model = TSNE(
            perplexity=tsne_perp, random_state=tsne_seed,
            init=tsne_init, **tsne_kwargs)

    Y = tsne_model.fit_transform(pc_scores.values)
    t1 = time.time()
    _LOGGER.info('t-SNE took %.1f s.' % (t1-t0))

    if exaggerated_tsne:
        dim_labels = ['t-SNE* dim. %d' % (l+1)
                      for l in range(Y.shape[1])]
    else:
        dim_labels = ['t-SNE dim. %d' % (l+1)
                      for l in range(Y.shape[1])]

    tsne_scores = pd.DataFrame(
        index=pc_scores.index, columns=dim_labels, data=Y)

    cell_labels = tsne_scores.index.to_series().str.split('_').apply(
        lambda x:x[0])

    cluster_order = ['Reference', 'Target']

    fig, legend_fig = plot_cells_random_order(
        tsne_scores, cell_labels=cell_labels,
        cluster_order=cluster_order,
        cluster_colors=cluster_colors, marker_size=marker_size)

    return fig, legend_fig, tsne_scores, pc_scores
