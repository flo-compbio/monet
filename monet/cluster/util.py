# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import Union

from ..core import ExpMatrix

import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)

Numeric = Union[int, float]


def get_overexpressed_genes(
        matrix: ExpMatrix, cell_labels: pd.Series,
        exp_thresh: float = 0.05, ignore_outliers: bool = True,
        num_genes: int = 20) -> pd.DataFrame:
    """Determine most over-expressed genes for each cluster."""

    # make sure matrix and cell_labels are aligned
    matrix = matrix.loc[:, cell_labels.index]

    if ignore_outliers:
        # ignore the cluster named "Outliers", if it exists
        sel = (cell_labels != 'Outliers')
        matrix = matrix.loc[:, sel]
        cell_labels = cell_labels.loc[sel]

    _LOGGER.info('Ignoring mean expression values below %.3f', exp_thresh)

    data = []

    # scale matrix
    matrix = matrix.scale()

    # determine fold-changes for all clusters
    vc = cell_labels.value_counts()
    clusters = vc.index.tolist()
    X = np.zeros((len(clusters), matrix.num_genes), dtype=np.float32)
    cluster_mean = ExpMatrix(genes=matrix.genes, cells=clusters, data=X.T)
    for l in clusters:
        sel = (cell_labels == l)
        cluster_mean.loc[:, l] = matrix.loc[:, sel].mean(axis=1)

    # in calculation of fold change,
    # ignore all expression values below exp_thresh
    thresh_cluster_mean = cluster_mean.copy()
    thresh_cluster_mean[thresh_cluster_mean < exp_thresh] = exp_thresh

    # calculate fold change relative to average of other clusters
    X = np.ones((len(clusters), matrix.num_genes), dtype=np.float32)
    fold_change = ExpMatrix(genes=matrix.genes, cells=clusters, data=X.T)
    for l in clusters:
        sel = (thresh_cluster_mean.cells != l)
        fold_change.loc[:, l] = thresh_cluster_mean.loc[:, l] / \
                (thresh_cluster_mean.loc[:, sel].mean(axis=1))

    markers = []
    for l in clusters:
        change = fold_change.loc[:, l].sort_values(ascending=False)
        change = change[:num_genes]

        # scale mean expression values to 10K transcripts
        mean = cluster_mean.loc[change.index, l]
        mean = (10000 / cluster_mean.loc[:, l].sum()) * mean

        cluster_index = [l] * num_genes
        gene_index = change.index
        index = pd.MultiIndex.from_arrays(
            [cluster_index, gene_index], names=['cluster', 'gene'])

        data = np.c_[change.values, mean.values]

        markers.append(
            pd.DataFrame(
                index=index,
                columns=['Fold change', 'Mean expression (TP10K)'],
                data=data))

    markers = pd.concat(markers, axis=0)

    #markers = markers.swaplevel(0, 1).sort_index(
    #    level=1, sort_remaining=False).swaplevel(0, 1)

    return markers


def get_variable_genes(
        matrix: ExpMatrix, cell_labels: pd.Series,
        genes_per_cluster: int = 20,
        ignore_outliers: bool = True) -> pd.DataFrame:
    """Determine most variable genes for each cluster."""

    # make sure matrix and cell_labels are aligned
    matrix = matrix.loc[:, cell_labels.index]

    if ignore_outliers:
        # ignore the cluster named "Outliers", if it exists
        sel = (cell_labels != 'Outliers')
        matrix = matrix.loc[:, sel]
        cell_labels = cell_labels.loc[sel]

    data = []

    vc = cell_labels.value_counts()
    all_clusters = vc.index.tolist()

    data = {}
    for cluster in all_clusters:
        sel = (cell_labels == cluster)
        var_genes = matrix.loc[:, sel].get_variable_genes(
            num_genes=genes_per_cluster)
        data[cluster] = var_genes

    var_genes = pd.concat(data, names=['Cluster', 'Gene'])

    return var_genes
