# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from typing import Tuple, Iterable
import logging

import plotly.graph_objs as go
import pandas as pd
import numpy as np

from ..core import ExpMatrix
from .. import util

_LOGGER = logging.getLogger(__name__)


def ma_plot(matrix, cluster1_cells, cluster2_cells, low_thresh=0.10,
            title=None) -> Tuple[go.Figure, pd.DataFrame]:

    #transcript_count = matrix.sum(axis=0)
    #scaled_matrix = (1e4 / transcript_count) * matrix

    transcript_count = matrix.median_transcript_count
    factor = 1e4 / transcript_count
    rel_low_thresh = low_thresh * factor

    scaled_matrix = matrix.scale()
    mean1 = scaled_matrix.loc[:, cluster1_cells].mean(axis=1)
    mean2 = scaled_matrix.loc[:, cluster2_cells].mean(axis=1)

    mean1 = util.scale(mean1, 10000)
    mean2 = util.scale(mean2, 10000)

    thresh1 = mean1.copy()
    thresh1.loc[thresh1 < rel_low_thresh] = rel_low_thresh

    thresh2 = mean2.copy()
    thresh2.loc[thresh2 < rel_low_thresh] = rel_low_thresh

    fc = thresh1 / thresh2

    mean = (mean1 + mean2)/2
    sel_genes = (mean >= rel_low_thresh)

    fc = fc.loc[sel_genes]
    mean = mean.loc[sel_genes]

    # create DataFrame with values
    d = {'mean': mean, 'fold change': fc}
    df = pd.DataFrame(d)

    x = mean.values
    y = np.log2(fc.values)
    text = mean.index.to_list()

    xlog = np.log10(x)
    ptp = np.ptp(xlog)
    #xrange = [pow(10, xlog.min() - 0.02*ptp), pow(10, xlog.max() + 0.02*ptp)]
    xrange = [xlog.min() - 0.025*ptp, xlog.max() + 0.025*ptp]

    bgline = go.Scatter(
        x=np.power(10, np.float64(xrange)),
        y=[0, 0],
        mode='lines',
        line=dict(width=1, color='lightgray'),
    )

    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        mode='markers',
        marker=dict(size=4, opacity=0.7, color='navy'),
    )

    data = [bgline, trace]

    layout = go.Layout(
        width=900,
        height=600,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(linecolor='black', ticks='outside', title='Mean expression level (TP10K)', type='log', dtick='D3', range=xrange),
        yaxis=dict(linecolor='black', ticks='outside', title='log<sub>2</sub>-Fold change'),
        title=title,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return fig, df


def ref_ma_plot(
        ref_matrix: ExpMatrix, comp_matrix: ExpMatrix,
        low_thresh=0.10, min_thresh = 0.01, title=None) \
            -> Tuple[go.Figure, pd.DataFrame]:
    """Asymmetrical MA plot to compare a treatment condition to a control.

    """

    gene_union = ref_matrix.genes | comp_matrix.genes
    ref_matrix = ref_matrix.reindex(gene_union).fillna(0)
    comp_matrix = comp_matrix.reindex(gene_union).fillna(0)

    ref_mean = ref_matrix.scale().mean(axis=1)
    comp_mean = comp_matrix.scale().mean(axis=1)

    ref_mean = util.scale(ref_mean, 10000)
    comp_mean = util.scale(comp_mean, 10000)

    #sel_genes = (ref_mean > low_thresh) | (comp_mean > low_thresh)
    sel_genes = (ref_mean >= min_thresh)
    _LOGGER.info('Kept %d genes.', sel_genes.sum())

    ref_mean = ref_mean.loc[sel_genes]
    comp_mean = comp_mean.loc[sel_genes]

    #ref_mean.loc[ref_mean < min_thresh] = min_thresh
    #comp_mean.loc[comp_mean < min_thresh] = min_thresh

    ref_thresh = ref_mean.copy()
    ref_thresh.loc[ref_thresh < low_thresh] = low_thresh

    comp_thresh = comp_mean.copy()
    comp_thresh.loc[comp_thresh < low_thresh] = low_thresh

    fc = comp_thresh / ref_thresh
    mean = ref_mean

    # create DataFrame with values
    d = {'mean': mean, 'fold change': fc}
    df = pd.DataFrame(d)    

    x = mean.values
    y = np.log2(fc.values)
    text = mean.index.to_list()

    xlog = np.log10(x)
    ptp = np.ptp(xlog)
    xrange = [xlog.min() - 0.025*ptp, xlog.max() + 0.025*ptp]

    bgline = go.Scatter(
        x=np.power(10, np.float64(xrange)),
        y=[0, 0],
        mode='lines',
        line=dict(width=1, color='lightgray'),
    )

    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        mode='markers',
        marker=dict(size=4, opacity=0.7, color='navy'),
    )

    data = [bgline, trace]

    layout = go.Layout(
        width=900,
        height=600,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(linecolor='black', ticks='outside',
                   title='Ref. expression level (TP10K)',
                   type='log', dtick='D3', range=xrange),
        yaxis=dict(linecolor='black', ticks='outside',
                   title='log<sub>2</sub>-Fold change'),
        title=title,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return fig, df


def diff_diff_plot(
        ref_matrix1: ExpMatrix, comp_matrix1: ExpMatrix,
        ref_matrix2: ExpMatrix, comp_matrix2: ExpMatrix,
        rep1_label = 'Replicate 1', rep2_label = 'Replicate 2',
        low_thresh=0.10, min_thresh = 0.01, title=None,
        highlight_genes: Iterable[str] = None) \
            -> Tuple[go.Figure, pd.Series, pd.Series]:
    """Diff-diff scatter plot.

    """

    gene_union = ref_matrix1.genes | comp_matrix1.genes | ref_matrix2.genes | comp_matrix2.genes

    ref_matrix1 = ref_matrix1.reindex(gene_union).fillna(0)
    comp_matrix1 = comp_matrix1.reindex(gene_union).fillna(0)
    ref_matrix2 = ref_matrix2.reindex(gene_union).fillna(0)
    comp_matrix2 = comp_matrix2.reindex(gene_union).fillna(0)

    ref_mean1 = ref_matrix1.scale().mean(axis=1)
    comp_mean1 = comp_matrix1.scale().mean(axis=1)
    ref_mean2 = ref_matrix2.scale().mean(axis=1)
    comp_mean2 = comp_matrix2.scale().mean(axis=1)

    ref_mean1 = util.scale(ref_mean1, 10000)
    comp_mean1 = util.scale(comp_mean1, 10000)
    ref_mean2 = util.scale(ref_mean2, 10000)
    comp_mean2 = util.scale(comp_mean2, 10000)

    sel_genes = (ref_mean1 > low_thresh) | (comp_mean1 > low_thresh) | (ref_mean2 > low_thresh) | (comp_mean2 > low_thresh)
    _LOGGER.info('Kept %d genes.', sel_genes.sum())

    ref_mean1 = ref_mean1.loc[sel_genes]
    comp_mean1 = comp_mean1.loc[sel_genes]
    ref_mean2 = ref_mean2.loc[sel_genes]
    comp_mean2 = comp_mean2.loc[sel_genes]

    ref_mean1.loc[ref_mean1 < min_thresh] = min_thresh
    comp_mean1.loc[comp_mean1 < min_thresh] = min_thresh
    ref_mean2.loc[ref_mean2 < min_thresh] = min_thresh
    comp_mean2.loc[comp_mean2 < min_thresh] = min_thresh

    ref_thresh1 = ref_mean1.copy()
    ref_thresh1.loc[ref_thresh1 < low_thresh] = low_thresh

    comp_thresh1 = comp_mean1.copy()
    comp_thresh1.loc[comp_thresh1 < low_thresh] = low_thresh

    ref_thresh2 = ref_mean2.copy()
    ref_thresh2.loc[ref_thresh2 < low_thresh] = low_thresh

    comp_thresh2 = comp_mean2.copy()
    comp_thresh2.loc[comp_thresh2 < low_thresh] = low_thresh

    fc1 = comp_thresh1 / ref_thresh1
    fc2 = comp_thresh2 / ref_thresh2

    x = np.log2(fc1.values)
    y = np.log2(fc2.values)
    text = fc1.index.to_list()

    #xlog = np.log10(x)
    #ylog = np.log10(y)
    xptp = np.ptp(x)
    yptp = np.ptp(y)

    xrange = [x.min() - 0.025*xptp, x.max() + 0.025*xptp]
    yrange = [y.min() - 0.025*yptp, y.max() + 0.025*yptp]
    axis_range = [min(xrange[0], yrange[0]), max(xrange[1], yrange[1])]

    bgline = go.Scatter(
        x=axis_range,
        y=axis_range,
        mode='lines',
        line=dict(width=2, color='lightgray'),
    )

    trace = go.Scatter(
        x=x,
        y=y,
        text=text,
        mode='markers',
        marker=dict(size=5, opacity=0.7, color='navy'),
    )

    data = [bgline, trace]

    if highlight_genes is not None:
        not_found = [g for g in highlight_genes if g not in fc1.index]
        if not_found:
            _LOGGER.warning('Highlight genes not found: %s',
                            ', '.join(not_found))
        highlight_genes = fc1.index & highlight_genes

        x = np.log2(fc1.loc[highlight_genes].values)
        y = np.log2(fc2.loc[highlight_genes].values)
        text = fc1.loc[highlight_genes].index.to_list()
        trace = go.Scatter(
            x=x,
            y=y,
            text=text,
            mode='markers',
            marker=dict(size=5, color='red'),
        )
        data.append(trace)

    layout = go.Layout(
        width=900,
        height=800,
        font=dict(family='serif', size=32),
        plot_bgcolor='white',
        xaxis=dict(linecolor='black', zerolinecolor='black', showline=False, zeroline=True, ticks='outside', title='%s log<sub>2</sub>-FC' % rep1_label),
        yaxis=dict(linecolor='black', zerolinecolor='black', showline=False, zeroline=True, ticks='outside', title='%s log<sub>2</sub>-FC' % rep2_label),
        title=title,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    return fig, fc1, fc2
