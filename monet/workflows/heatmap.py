# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Iterable, List, Tuple, Dict, Union
import gc
import logging

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..visualize import Heatmap, HeatmapPanel, HeatmapAnnotation, HeatmapLayout
from ..cluster import order_matrix, get_overexpressed_genes, get_variable_genes
from ..util import get_variable_genes_cv, get_variable_genes_fano
from ..visualize.util import DEFAULT_PLOTLY_COLORS, DEFAULT_GGPLOT_COLORS
from .. import util
from .util import get_default_cluster_colors

_LOGGER = logging.getLogger(__name__)

Numeric = Union[float, int]
ExpData = Union[str, ExpMatrix]
LabelData = Union[str, pd.Series]


def overexpressed_gene_heatmap(
        raw_matrix: ExpData,
        denoised_matrix: ExpData,
        cell_labels: LabelData,
        genes_per_cluster: int = 20,
        exp_thresh: float = 0.05, **kwargs) \
            -> Tuple[go.Figure, go.Figure, go.Figure, ExpMatrix, ExpMatrix]:

    if isinstance(raw_matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(raw_matrix)
    else:
        matrix = raw_matrix

    if isinstance(cell_labels, str):
        cell_labels_loaded = util.load_cell_labels(cell_labels)
    else:
        cell_labels_loaded = cell_labels

    markers = get_overexpressed_genes(
        matrix, cell_labels_loaded, exp_thresh=exp_thresh,
        num_genes=genes_per_cluster)

    var_genes = []
    for cluster in markers.index.unique(level='cluster'):
        for gene, vals in markers.loc[cluster].iterrows():
            var_genes.append(gene)

    var_genes = list(dict.fromkeys(var_genes))
    _LOGGER.info('Total number of overexpressed genes: %d', len(var_genes))

    var_label = kwargs.pop('var_label', None)
    if var_label is None:
        var_label = 'Overexpressed<br>genes (%d)' % len(var_genes)

    denoised_heatmap_fig, raw_heatmap_fig, marker_matrix, var_matrix = \
        singlecell_heatmap(
            raw_matrix, denoised_matrix, cell_labels,
            var_genes=var_genes, var_label=var_label, **kwargs)

    return markers, denoised_heatmap_fig, raw_heatmap_fig, \
            marker_matrix, var_matrix


def variable_gene_heatmap(
        raw_matrix: ExpData,
        denoised_matrix: ExpData,
        cell_labels: LabelData,
        num_genes: int = 500,
        min_exp_thresh: float=0.10,
        sel_cluster: str = None,
        **kwargs) \
            -> Tuple[go.Figure, go.Figure, go.Figure, ExpMatrix, ExpMatrix]:

    if isinstance(denoised_matrix, str):
        # treat as file path
        matrix = ExpMatrix.load_enhance(denoised_matrix)
    else:
        matrix = denoised_matrix

    if isinstance(cell_labels, str):
        cell_labels_loaded = util.load_cell_labels(cell_labels)
    else:
        cell_labels_loaded = cell_labels

    if sel_cluster is not None:
        if isinstance(sel_cluster, (str, int)):
            sel_cluster = [sel_cluster]
        cluster_labels = kwargs.get('cluster_labels', {})
        mapped_cell_labels = cell_labels_loaded.replace(cluster_labels)
        mapped_sel_cluster = []
        for cluster in sel_cluster:
            try:
                mapped_cluster = cluster_labels[cluster]
            except KeyError:
                mapped_sel_cluster.append(cluster)
            else:
                mapped_sel_cluster.append(mapped_cluster)

        sel_cells = (mapped_cell_labels.isin(mapped_sel_cluster))
        cell_labels_loaded = cell_labels_loaded.loc[sel_cells]
        matrix = matrix.loc[:, sel_cells]

    var_genes_df, _ = get_variable_genes_cv(
        matrix, num_genes, min_exp_thresh=min_exp_thresh)

    var_genes = var_genes_df.index.tolist()
    _LOGGER.info('Total number of variable genes: %d', len(var_genes))

    var_label = kwargs.pop('var_label', None)
    if var_label is None:
        var_label = 'Variable genes (%d)' % len(var_genes)

    denoised_heatmap_fig, raw_heatmap_fig, marker_matrix, var_matrix = \
        singlecell_heatmap(
            raw_matrix, denoised_matrix, cell_labels_loaded,
            var_genes=var_genes, var_label=var_label, **kwargs)

    return var_genes, denoised_heatmap_fig, raw_heatmap_fig, \
            marker_matrix, var_matrix


def variable_gene_heatmap_old(
        raw_matrix: ExpData,
        denoised_matrix: ExpData,
        cell_labels: LabelData,
        genes_per_cluster: int = 20,
        select_cluster: str = None,
        **kwargs) \
            -> Tuple[go.Figure, go.Figure, go.Figure, ExpMatrix, ExpMatrix]:

    if isinstance(raw_matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(raw_matrix)
    else:
        matrix = raw_matrix

    if isinstance(cell_labels, str):
        cell_labels_loaded = util.load_cell_labels(cell_labels)
    else:
        cell_labels_loaded = cell_labels

    if select_cluster is not None:
        cluster_labels = kwargs.get('cluster_labels', {})
        mapped_cell_labels = cell_labels_loaded.replace(cluster_labels)
        try:
            mapped_select_cluster = cluster_labels[select_cluster]
        except KeyError:
            mapped_select_cluster = select_cluster

        sel_cells = (mapped_cell_labels == mapped_select_cluster)
        cell_labels_loaded = cell_labels_loaded.loc[sel_cells]
        matrix = matrix.loc[:, sel_cells]

    cluster_var_genes = get_variable_genes(
        matrix, cell_labels_loaded, genes_per_cluster)

    var_genes = []
    for cluster in cluster_var_genes.index.unique(level='Cluster'):
        for gene, vals in cluster_var_genes.loc[cluster].iterrows():
            var_genes.append(gene)

    var_genes = list(dict.fromkeys(var_genes))
    _LOGGER.info('Total number of variable genes: %d', len(var_genes))

    var_label = kwargs.pop('var_label', None)
    if var_label is None:
        var_label = 'Variable genes (%d)' % len(var_genes)

    denoised_heatmap_fig, raw_heatmap_fig, marker_matrix, var_matrix = \
        singlecell_heatmap(
            raw_matrix, denoised_matrix, cell_labels_loaded,
            var_genes=var_genes, var_label=var_label, **kwargs)

    return var_genes, denoised_heatmap_fig, raw_heatmap_fig, \
            marker_matrix, var_matrix


def singlecell_heatmap(
        raw_matrix: ExpData,
        denoised_matrix: ExpData,
        cell_labels: LabelData = None,
        marker_genes: Iterable[str] = None, cluster_marker_genes: bool = True,
        marker_height: float = 0.15,
        marker_label: str = None,
        markers_always_raw: bool = False,
        var_genes: Iterable[str] = None, cluster_var_genes: bool = True,
        var_label: str = None,
        use_zscores: bool = True,
        cell_order: Iterable[str] = None,
        cluster_order: List[str] = None,
        cluster_colors: Dict[str, str] = None,
        cluster_labels: Dict[str, str] = None,
        title: str = None, width: int = 1100, height: int = 850,
        annotation_height: int = 30,
        zmin: float = -3.0, zmax: float = 3.0,
        max_cells: int = 2000, seed: int = 0,
        annotation_label: str = 'Clustering',
        colorbar_label: str = None,
        colorscheme: str = 'plotly') -> \
            Tuple[go.Figure, go.Figure, ExpMatrix, ExpMatrix]:

    if colorbar_label is None and use_zscores:
        colorbar_label = 'z-score'

    if cluster_colors is None:
        cluster_colors = {}

    if cluster_labels is None:
        cluster_labels = {}

    if marker_height <= 0 or marker_height >= 1.0:
        raise ValueError('Marker height must be between 0 and 1.')

    if isinstance(cell_labels, str):
        cell_labels = util.load_cell_labels(cell_labels)

    if isinstance(denoised_matrix, str):
        # treat as file path
        matrix = ExpMatrix.load_enhance(denoised_matrix)
    else:
        matrix = denoised_matrix

    # align denoised matrix with labels
    if cell_labels is not None:
        matrix = matrix.loc[:, cell_labels.index]

    if cell_labels is None:
        # create a dummy cluster
        cell_labels_ = pd.Series(index=matrix.cells, data=[0]*matrix.num_cells)
    else:
        cell_labels_ = cell_labels

    num_clusters = cell_labels_.value_counts().size

    # check if any of the variable genes provided are missing
    var_genes = pd.Index(var_genes)
    is_unknown = ~var_genes.isin(matrix.genes)
    if is_unknown.any():
        _LOGGER.warning(
            '%d / %d variable genes not contained in expression matrix: '
            '%s', is_unknown.sum(), var_genes.size,
            ', '.join(var_genes[is_unknown]))
    var_genes = var_genes[~is_unknown]
    if var_label is None:
        var_label = 'Genes (%d)' % len(var_genes)

    # convert matrix to z-score
    if use_zscores:
        np.seterr(invalid='ignore')
        matrix = matrix.scale().standardize_genes().fillna(0)
        np.seterr(invalid='warn')

    # get differential gene matrix and cluster genes
    var_matrix = matrix.reindex(var_genes).fillna(0)
    if cluster_var_genes:
        var_matrix = order_matrix(var_matrix, axis='both')
    else:
        var_matrix = order_matrix(var_matrix, axis='cells')

    if cell_order is None:

        if cluster_order is None:
            # if no cluster order is provided,
            # perform clustering on cluster averages
            # using the overexpressed genes
            cluster_matrix = var_matrix.groupby(cell_labels_, axis=1).mean()
            num_clusters = cluster_matrix.shape[1]
            cluster_matrix = ExpMatrix(cluster_matrix)
            if num_clusters > 1:
                cluster_matrix = order_matrix(cluster_matrix, axis='cells')
            cluster_order = cluster_matrix.cells.copy()

        else:
            # if clusters are missing in the cluster order provided, add them,
            # largest first
            all_clusters = cell_labels_.value_counts().index.tolist()
            for cluster in all_clusters:
                if cluster not in cluster_order:
                    cluster_order.append(cluster)

        # group cells according to cluster first
        label_mapping = dict([c, i] for i, c in enumerate(cluster_order))
        cell_labels_ = cell_labels_.loc[var_matrix.cells]
        numeric_cell_labels = cell_labels_.loc[var_matrix.cells].map(label_mapping)
        a = np.lexsort([np.arange(var_matrix.num_cells), numeric_cell_labels])
        var_matrix = var_matrix.iloc[:, a]

        # align cell_labels with cells in var_gene matrix
        cell_labels_ = cell_labels_.loc[var_matrix.cells]

    else:
        # use provided cell order
        var_matrix = var_matrix.loc[:, cell_order]
        cell_labels_ = cell_labels_.loc[var_matrix.cells]

        if cluster_order is None:
            cluster_order = cell_labels_.loc[~cell_labels_.duplicated()].values.tolist()

    if marker_genes is not None:
        # get marker matrix
        marker_genes = pd.Index(marker_genes)
        is_unknown = ~marker_genes.isin(matrix.genes)
        if is_unknown.any():
            _LOGGER.warning(
                '%d / %d marker genes not contained in expression matrix: %s',
                is_unknown.sum(), marker_genes.size,
                ', '.join(marker_genes[is_unknown]))
        marker_genes = marker_genes[~is_unknown]
        marker_matrix = matrix.reindex(marker_genes).fillna(0)
        if cluster_marker_genes:
            marker_matrix = order_matrix(marker_matrix, axis='genes')

        # align cells in marker matrix with cells in var_gene matrix
        marker_matrix = marker_matrix.loc[:, var_matrix.cells]

        if marker_label is None:
            marker_label = 'Marker<br>genes (%d)' % marker_matrix.shape[0]

    else:
        marker_height = 0.0
        marker_matrix = None

    # delete expression matrix and free memory if possible
    del matrix; gc.collect()

    num_clusters = cell_labels_.value_counts().size
    if marker_genes is None:
        exp_colorbar_index = 0
    else:
        exp_colorbar_index = 1
    if cell_labels is not None:
        exp_colorbar_index += 1

    # select which cells to display
    # => cell_labels
    # => marker_matrix
    # => var_matrix

    if max_cells < var_matrix.num_cells:
        _LOGGER.info('Randomly sampling %d / %d cells while maintaining '
                     'cluster proportions', max_cells, var_matrix.num_cells)
        var_matrix_disp = util.sample_matrix(
            var_matrix, max_cells, cell_labels_, seed=seed)
        if marker_genes is not None:
            marker_matrix_disp = marker_matrix.loc[:, var_matrix_disp.cells]
        cell_labels_disp = cell_labels_.loc[var_matrix_disp.cells]

    else:
        var_matrix_disp = var_matrix
        marker_matrix_disp = marker_matrix
        cell_labels_disp = cell_labels_

    ### plot same heatmap for raw data

    if isinstance(raw_matrix, str):
        # treat as file path
        matrix = ExpMatrix.load(raw_matrix)
    else:
        matrix = raw_matrix

    # align raw matrix with labels
    matrix = matrix.loc[:, cell_labels_.index]

    # convert to z-score
    if use_zscores:
        np.seterr(invalid='ignore')
        matrix = matrix.scale().standardize_genes().fillna(0)
        np.seterr(invalid='warn')

    var_matrix2 = matrix.reindex(var_matrix.genes).fillna(0)

    if marker_genes is not None:
        marker_matrix2 = matrix.reindex(marker_matrix.genes).fillna(0)
        marker_matrix2_disp = marker_matrix2.loc[:, cell_labels_disp.index]
    else:
        marker_matrix2 = None
        marker_matrix2_disp = None

    # delete expression matrix and free memory
    del matrix; gc.collect()

    var_matrix2_disp = var_matrix2.loc[:, cell_labels_disp.index]

    ticktext = []
    for c in reversed(cluster_order):
        try:
            ticktext.append(cluster_labels[c])
        except KeyError:
            ticktext.append(c)

    ### generate denoised heatmap
    data = []

    if markers_always_raw:
        marker_matrix_disp = marker_matrix2_disp

    if title is not None:
        final_title = '%s (denoised data)' % title
    else:
        final_title = '(denoised data)'

    if cell_labels is not None:
        trace = HeatmapAnnotation(
            height=annotation_height,
            labels=cell_labels_disp,
            clusterorder=cluster_order,
            clustercolors=cluster_colors,
            clusterlabels=cluster_labels,
            title=annotation_label,
            colorscheme=colorscheme,
            showscale=True,
        )
        data.append(trace)

    if marker_genes is not None:
        trace = HeatmapPanel(
            height=marker_height,
            matrix=marker_matrix_disp,
            colorscale='RdBu',
            reversescale=True,
            title=marker_label,
            zmin=zmin, zmax=zmax,
            showscale=False)
        data.append(trace)

    trace = HeatmapPanel(
        height=0.95 - marker_height,
        matrix=var_matrix_disp,
        colorscale='RdBu',
        reversescale=True,
        title=var_label,
        zmin=zmin, zmax=zmax,
        showscale=True,
        colorbarlabel=colorbar_label)
    data.append(trace)

    layout = HeatmapLayout(
        title=final_title,
        height=height,
        width=width,
    )

    heatmap = Heatmap(data=data, layout=layout)
    fig = heatmap.get_figure()

    #fig.layout.xaxis.showticklabels=True
    #fig.layout.xaxis.ticks = 'outside'
    fig.layout.xaxis.title = None
    fig.layout.xaxis.side = 'top'
    fig.layout.xaxis.anchor = 'y'
    #fig.layout.margin.t = 200

    if cell_labels is not None:
        colorbar = dict(
            len=25*num_clusters,
            lenmode='pixels',
            x=1.01,
            xanchor='left',
            y=1.01,
            xpad=10,
            ypad=10,
            yanchor='top',
            tickvals=np.arange(len(cluster_order))+0.5,
            ticktext=ticktext,
            tickfont=dict(size=16),
            ticks=None,
            borderwidth=0,
        )
        fig.data[0].colorbar.update(colorbar)

    colorbar = dict(
        x=1.01,
        xanchor='left',
        y=0.3,
        yanchor='top',
        xpad=10,
        ypad=10,
        len=120,
        lenmode='pixels',
    )
    fig.data[exp_colorbar_index].colorbar.update(colorbar)

    denoised_fig = fig

    # generate raw heatmap
    data = []

    if title is not None:
        final_title = '%s (raw data)' % title
    else:
        final_title = '(raw data)'

    if cell_labels is not None:
        trace = HeatmapAnnotation(
            height=annotation_height,
            labels=cell_labels_disp,
            clusterorder=cluster_order,
            clustercolors=cluster_colors,
            clusterlabels=cluster_labels,
            title=annotation_label,
            colorscheme=colorscheme,
            showscale=True,
        )
        data.append(trace)

    if marker_genes is not None:
        trace = HeatmapPanel(
            height=marker_height,
            matrix=marker_matrix2_disp,
            colorscale='RdBu',
            reversescale=True,
            title=marker_label,
            zmin=zmin, zmax=zmax,
            showscale=False)
        data.append(trace)

    trace = HeatmapPanel(
        height=0.95 - marker_height,
        matrix=var_matrix2_disp,
        colorscale='RdBu',
        reversescale=True,
        title=var_label,
        zmin=zmin, zmax=zmax,
        showscale=True,
        colorbarlabel=colorbar_label)        
    data.append(trace)

    layout = HeatmapLayout(
        title=final_title,
        height=height,
        width=width,
    )

    heatmap = Heatmap(data=data, layout=layout)
    fig = heatmap.get_figure()

    #fig.layout.xaxis.showticklabels=True
    #fig.layout.xaxis.ticks = 'outside'
    fig.layout.xaxis.title = None
    fig.layout.xaxis.side = 'top'
    fig.layout.xaxis.anchor = 'y'
    #fig.layout.margin.t = 200

    if cell_labels is not None:
        colorbar = dict(
            len=25*num_clusters,
            lenmode='pixels',
            xanchor='left',
            y=1.01,
            xpad=10,
            ypad=10,
            yanchor='top',
            tickvals=np.arange(len(cluster_order))+0.5,
            ticktext=ticktext,
            tickfont=dict(size=16),
            ticks=None,
            borderwidth=0,
        )
        fig.data[0].colorbar.update(colorbar)

    colorbar = dict(
        x=1.01,
        xanchor='left',
        y=0.3,
        yanchor='top',
        xpad=10,
        ypad=10,
        len=120,
        lenmode='pixels',
    )
    fig.data[exp_colorbar_index].colorbar.update(colorbar)

    raw_fig = fig

    return denoised_fig, raw_fig, marker_matrix, var_matrix
