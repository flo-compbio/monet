# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from typing import Dict, Tuple, Iterable, Union
import logging
import sys

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix

_LOGGER = logging.getLogger(__name__)

Numeric = Union[float, int]


def sample_cells(matrix: ExpMatrix, num_cells=1000,
                 seed: int = 0) -> ExpMatrix:
    """Sample cells from a matrix without altering the relative cell order."""

    if num_cells > matrix.num_cells:
        raise ValueError('Cannot sample %d cells from a matrix that only contains %d cells.',
                         num_cells, matrix.num_cells)

    np.random.seed(seed)
    sel = np.random.choice(matrix.num_cells, num_cells, replace=False)
    sel.sort()
    sel_matrix = matrix.iloc[:, sel]

    return sel_matrix


def combine_matrices(datasets: Dict[str, ExpMatrix]) \
        -> Tuple[ExpMatrix, pd.Series]:
    """Combine multiple expression matrices (samples)."""

    # prepand dataset name to cell names for each dataset
    for name, matrix in datasets.items():
        matrix.cells = matrix.cells.to_series().apply(
            lambda x: '%s_%s' % (name, x)).values

    combined_matrix = pd.concat(datasets.values(), axis=1).fillna(0)

    cell_labels = combined_matrix.cells.to_series().str.split('_').apply(
        lambda x: x[0])

    return combined_matrix, cell_labels


def convert_genes(matrix: ExpMatrix, mapping: pd.Series,
                  keep_unknown: bool = False) -> ExpMatrix:
    """Convert gene identifiers."""

    num_genes = matrix.shape[0]
    num_unknown = (~matrix.genes.isin(mapping.index)).sum()

    if keep_unknown:
        # keep genes with unknown identifiers
        conv_matrix = matrix.copy()
        conv_matrix.genes = conv_matrix.genes.replace(mapping)
        _LOGGER.info(
            'Ignored %d / %d genes with unknown identifiers (%.1f %%).',
            num_unknown, num_genes, 100 * (num_unknown / num_genes))

    else:
        # remove genes with unknown identifiers
        conv_matrix = matrix.loc[matrix.genes.isin(mapping.index)]
        conv_matrix.genes = conv_matrix.genes.map(mapping)
        _LOGGER.info(
            'Removed %d / %d genes with unknown identifiers (%.1f %%).',
            num_unknown, num_genes, 100 * (num_unknown / num_genes))

    num_dups = conv_matrix.genes.duplicated().sum()
    _LOGGER.info('Number of duplicate gene entries: %d', num_dups)

    if num_dups > 0:
        _LOGGER.info('Aggregating duplicate genes...'); sys.stdout.flush()
        conv_matrix = ExpMatrix(conv_matrix.groupby(conv_matrix.genes).sum())

    return conv_matrix


def get_variable_genes_cv(
        matrix: ExpMatrix,
        num_genes: int = 1000,
        min_exp_thresh: Numeric = 0.05,
        sel_genes: Iterable[str] = None,
        marker_size: Numeric = 3) -> Tuple[pd.DataFrame, go.Figure]:
    """Select most variable genes based on coefficient of variation."""

    scaled_matrix = matrix.scale()

    if sel_genes is not None:
        scaled_matrix = scaled_matrix.loc[sel_genes]

    # remove genes that are not expressed
    scaled_matrix = scaled_matrix.loc[scaled_matrix.sum(axis=1) > 0]

    # calculate tp10k mean for plotting later
    mean = scaled_matrix.mean(axis=1)
    tp10k_mean = (1e4 / mean.sum()) * mean

    # apply expression threshold
    if min_exp_thresh > 0:
        scaled_matrix.values[scaled_matrix.values < min_exp_thresh] = \
                min_exp_thresh

    # calculate mean and standard deviation of each gene
    mean = scaled_matrix.mean(axis=1)
    std = scaled_matrix.std(axis=1, ddof=1)

    cv = std / mean

    index = cv.sort_values(ascending=False).iloc[:num_genes].index
    data = data = np.c_[cv.loc[index].values, tp10k_mean.loc[index].values]
    columns = ['CV', 'Mean expression (TP10K)']
    var_genes = pd.DataFrame(data, index=index, columns=columns).copy()

    selected = cv.index.isin(var_genes.index)

    data = []

    # first, plot genes that *weren't* selected
    x = tp10k_mean.loc[~selected].values
    y = 100*cv.loc[~selected].values
    text = mean.index[~selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Not selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    # next, plot genes that *were* selected
    x = tp10k_mean.loc[selected].values
    y = 100*cv.loc[selected].values
    text = mean.index[selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    yaxis_title = 'CV (%)'
    layout = go.Layout(
        width=800,
        height=800,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(title='Mean expression (TP10K)', linecolor='black',
                   ticks='outside', ticklen=5, type='log', dtick='D3'),
        #yaxis=dict(title='Fano factor', linecolor='black', ticks='outside', ticklen=5),
        yaxis=dict(title=yaxis_title, linecolor='black', ticks='outside',
                   ticklen=5, type='log', dtick='D3'),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return var_genes, fig


def get_variable_genes_fano(
        matrix: ExpMatrix,
        num_genes: int = 1000,
        min_exp_thresh: Numeric = 0,
        sel_genes: Iterable[str] = None,
        marker_size: Numeric = 3) -> Tuple[pd.DataFrame, go.Figure]:
    """Select most variable genes based on Fano factor."""

    scaled_matrix = matrix.scale()

    if sel_genes is not None:
        scaled_matrix = scaled_matrix.loc[sel_genes]

    mean = scaled_matrix.mean(axis=1)
    var = scaled_matrix.var(axis=1, ddof=1)

    # remove constant genes
    var = var.loc[var > 0]
    mean = mean.loc[var.index]

    # calculate scaled mean for plotting later
    scaled_mean = (1e4 / mean.sum()) * mean

    # apply expression threshold
    if min_exp_thresh > 0:
        mean.loc[mean < min_exp_thresh] = min_exp_thresh

    fano = var / mean

    index = fano.sort_values(ascending=False).iloc[:num_genes].index
    data = data = np.c_[fano.loc[index].values, scaled_mean.loc[index].values]
    columns = ['Fano factor', 'Mean expression (TP10K)']
    var_genes = pd.DataFrame(data, index=index, columns=columns).copy()

    selected = fano.index.isin(var_genes.index)

    data = []

    # first, plot genes that *weren't* selected
    x = scaled_mean.loc[~selected].values
    y = fano.loc[~selected].values
    text = mean.index[~selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Not selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    # next, plot genes that *were* selected
    x = scaled_mean.loc[selected].values
    y = fano.loc[selected].values
    text = mean.index[selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    if min_exp_thresh == 0:
        yaxis_title = 'Fano factor'
    else:
        yaxis_title = 'Modified Fano factor'

    layout = go.Layout(
        width=800,
        height=800,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(title='Mean expression (TP10K)', linecolor='black',
                   ticks='outside', ticklen=5, type='log', dtick='D3'),
        #yaxis=dict(title='Fano factor', linecolor='black', ticks='outside', ticklen=5),
        yaxis=dict(title=yaxis_title, linecolor='black', ticks='outside',
                   ticklen=5, type='log', dtick='D3'),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return var_genes, fig


def get_variable_genes(
        matrix: ExpMatrix,
        num_genes: int = 1000,
        thresh: Numeric = 100,
        sel_genes: Iterable[str] = None,
        marker_size: Numeric = 3) -> Tuple[pd.DataFrame, go.Figure]:
    """Select most variable genes based on coefficient of variation."""

    scaled_matrix = matrix.scale()

    if sel_genes is not None:
        scaled_matrix = scaled_matrix.loc[sel_genes]

    # remove genes without expression
    scaled_matrix = scaled_matrix.loc[scaled_matrix.mean(axis=1) > 0]

    # calculate tp10k mean for plotting later
    mean = scaled_matrix.mean(axis=1)
    tp10k_mean = (1e4 / mean.sum()) * mean

    

    # calculate z-scores and apply threshold
    zscore_matrix = scaled_matrix.standardize_genes()
    zscore_matrix.values[zscore_matrix.values < zscore_thresh] = zscore_thresh

    # calculate variance score
    var_score = zscore_matrix.sum(axis=1)

    index = var_score.sort_values(ascending=False).iloc[:num_genes].index
    data = data = np.c_[var_score.loc[index].values, tp10k_mean.loc[index].values]
    columns = ['CV', 'Mean expression (TP10K)']
    var_genes = pd.DataFrame(data, index=index, columns=columns).copy()

    selected = cv.index.isin(var_genes.index)

    data = []

    # first, plot genes that *weren't* selected
    x = tp10k_mean.loc[~selected].values
    y = 100*cv.loc[~selected].values
    text = mean.index[~selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Not selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    # next, plot genes that *were* selected
    x = tp10k_mean.loc[selected].values
    y = 100*cv.loc[selected].values
    text = mean.index[selected].to_list()
    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        name='Selected',
        mode='markers',
        marker=dict(size=marker_size, opacity=0.7),
    )
    data.append(trace)

    yaxis_title = 'CV (%)'
    layout = go.Layout(
        width=800,
        height=800,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(title='Mean expression (TP10K)', linecolor='black',
                   ticks='outside', ticklen=5, type='log', dtick='D3'),
        #yaxis=dict(title='Fano factor', linecolor='black', ticks='outside', ticklen=5),
        yaxis=dict(title=yaxis_title, linecolor='black', ticks='outside',
                   ticklen=5, type='log', dtick='D3'),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return var_genes, fig
