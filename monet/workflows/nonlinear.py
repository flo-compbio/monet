# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Tuple, Union, Dict

import pandas as pd
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..visualize import tsne_plot, umap_plot
from .. import util
from .util import get_default_cluster_colors

Numeric = Union[int, float]


def tsne(matrix: Union[str, ExpMatrix],
         cell_labels: Union[str, ExpMatrix] = None,
         num_components: int = 50,
         perplexity: Numeric = 30,
         cluster_colors: Dict[str, str] = None,
         **kwargs) -> Tuple[go.Figure, pd.DataFrame]:
    """Perform t-SNE."""

    if isinstance(matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(matrix)

    if cell_labels is not None:
        if isinstance(cell_labels, str):
            # treat as file path
            cell_labels = util.load_cell_labels(cell_labels)
        matrix = matrix.loc[:, cell_labels.index]

    if cluster_colors is None:
        cluster_colors = get_default_cluster_colors()

    fig, tsne_scores = tsne_plot(
        matrix,
        num_components=num_components,
        perplexity=perplexity,
        cell_labels=cell_labels,
        cluster_colors=cluster_colors,
        **kwargs)

    return fig, tsne_scores


def umap(matrix: Union[str, ExpMatrix],
         cell_labels: Union[str, pd.Series] = None,
         num_components: int = 50,
         num_neighbors: int = 30, min_dist: float = 0.3,
         cluster_colors: Dict[str, str] = None,
         **kwargs) -> Tuple[go.Figure, pd.DataFrame]:
    """Perform UMAP."""

    if isinstance(matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(matrix)

    if cell_labels is not None:
        if isinstance(cell_labels, str):
            # treat as file path
            cell_labels = util.load_cell_labels(cell_labels)
        matrix = matrix.loc[:, cell_labels.index]

    if cluster_colors is None:
        cluster_colors = get_default_cluster_colors()

    fig, umap_scores = umap_plot(
        matrix,
        num_components=num_components,
        num_neighbors=num_neighbors,
        min_dist=min_dist,
        cell_labels=cell_labels,
        cluster_colors=cluster_colors,
        **kwargs)

    return fig, umap_scores
