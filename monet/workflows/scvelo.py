# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Union

import pandas as pd

from ..velocyto import VelocytoData
from ..visualize import plot_cells, umap_plot
from ..velocyto import anndata_ft_transform
from ..core import ExpMatrix
from .. import util


def scvelo(data: Union[VelocytoData, str],
           cell_labels: Union[pd.Series, str] = None,
           num_components: int = 50, num_neighbors: int = 30,
           cluster_order = None, cluster_colors = None,
           cluster_labels = None,
           umap_scores: pd.DataFrame = None, umap_num_neighbors: int = 30,
           umap_seed: int = 0,
           title: str = None) -> None:
    """Modified scVelo workflow."""

    import scvelo as scv

    scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
    scv.settings.presenter_view = True  # set max width size for presenter view
    scv.set_figure_params('scvelo')  # for beautified visualization

    if isinstance(data, str):
        data = VelocytoData.load_npz(data)

    if isinstance(cell_labels, str):
        cell_labels = util.load_cell_labels(cell_labels)

    adata = data.to_anndata(
        cell_labels=cell_labels,
        cluster_labels=cluster_labels,
        cluster_order=cluster_order,
        cluster_colors=cluster_colors,
        umap_scores=umap_scores)

    try:
        clusters = adata.obs['clusters']
    except KeyError:
        clusters = None

    if 'X_umap' not in adata.obsm:
        # perform UMAP
        exp_matrix = ExpMatrix(
            adata.X.T, genes=adata.var_names, cells=adata.obs_names)
        _, umap_scores = umap_plot(
            exp_matrix, num_components, umap_num_neighbors, seed=umap_seed)
        adata.obsm['X_umap'] = umap_scores.values

    else:
        # use UMAP result stored in AnnData object
        columns = ['UMAP dim. 1', 'UMAP dim. 2']
        umap_scores = pd.DataFrame(
            adata.obsm['X_umap'], index=adata.obs_names, columns=columns)

    if clusters is not None:
        # plot UMAP with cell type labels
        cluster_order = clusters.cat.categories
        cluster_colors = adata.uns['plotly_cluster_colors']
        fig = plot_cells(
            umap_scores, cell_labels=clusters,
            cluster_order=cluster_order,
            cluster_colors=cluster_colors, labelsize=16, showlabels=False,
            title=title)
        color = None

    else:
        # plot UMAP without cell type labels
        fig = plot_cells(umap_scores, title=title)
        color = 'navy'

    fig.show()

    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    anndata_ft_transform(adata)

    scv.pp.moments(adata, n_pcs=num_components, n_neighbors=num_neighbors)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

    # with raw=True (default), pl.proporitons uses raw counts
    # in adata.obs['initial_size_spliced'] and adata.obs['initial_size_unspliced']
    # that were stored there by the pp.filter_genes() function
    scv.pl.proportions(adata, figsize=(8, 4))

    scv.pl.velocity_embedding_grid(
        adata, basis='umap', dpi=200, size=10,
        arrow_size=5, arrow_length=1.0, arrow_color='black',
        fontsize=6, linewidth=0.1,
        legend_fontsize=6,
        figsize=[4.5, 4.5],
        legend_loc='right margin',
        title=title)

    scv.pl.velocity_embedding_stream(
        adata, basis='umap', dpi=200, size=10,
        fontsize=6, linewidth=1.0,
        legend_fontsize=6,
        figsize=[4.5, 4.5],
        color=color,
        title=title)

    return adata
