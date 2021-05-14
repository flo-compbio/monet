# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Tuple, Union

import pandas as pd
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..cluster import cluster_cells_dbscan, cluster_cells_leiden
from ..visualize import tsne_plot, umap_plot, plot_cells
from .. import util


def density_based_clustering(
        matrix: Union[str, ExpMatrix], output_file: str = None,
        num_components: int = 50,
        min_cells_frac: float = 0.01, eps_frac: float = 0.03,
        tsne_scores: pd.DataFrame = None,
        title: str = None) \
            -> Tuple[pd.Series, go.Figure, go.Figure, pd.DataFrame]:
    """Perform density-based clustering using DBSCAN on t-SNE results."""

    if isinstance(matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(matrix)

    if tsne_scores is None:
        tsne_fig, tsne_scores = tsne_plot(
            matrix, num_components=num_components,
            title=title)

    else:
        tsne_fig = plot_cells(tsne_scores, title=title)

    tsne_fig.data[0].marker.color = 'navy'

    cell_labels, clusters = cluster_cells_dbscan(
        tsne_scores, min_cells_frac=min_cells_frac, eps_frac=eps_frac)

    cluster_colors = {
        'Outliers': 'lightgray',
    }

    cluster_fig = plot_cells(
        tsne_scores, cell_labels=cell_labels,
        cluster_order=clusters,
        cluster_colors=cluster_colors,
        title=title)

    if output_file is not None:
        util.save_cell_labels(cell_labels, output_file)

    return cell_labels, tsne_fig, cluster_fig, tsne_scores


def graph_based_clustering(
        matrix: Union[str, ExpMatrix], output_file: str = None,
        num_components: int = 50,
        resolution: float = 0.8,
        umap_kwargs: dict = None,
        title: str = None,
        umap_scores: pd.DataFrame = None) -> pd.Series:

    if isinstance(matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(matrix)

    if umap_kwargs is None:
        umap_kwargs = {}

    cell_labels, pca_model = cluster_cells_leiden(
        matrix,
        num_components=num_components,
        resolution=resolution)

    num_clusters = cell_labels.value_counts().size
    cluster_order = list(range(num_clusters))

    if umap_scores is None:
        umap_fig, umap_scores = umap_plot(
            matrix, num_components=num_components,
            title=title, **umap_kwargs)

    else:
        umap_fig = plot_cells(umap_scores, title=title)

    cluster_fig = plot_cells(
        umap_scores,
        cell_labels=cell_labels,
        cluster_order=cluster_order, colorscheme='ggplot',
        title=title)

    if output_file is not None:
        util.save_cell_labels(cell_labels, output_file)

    return cell_labels, umap_fig, cluster_fig, umap_scores
