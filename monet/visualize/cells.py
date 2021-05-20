# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import List, Dict, Union, Tuple
import sys
import time

from sklearn.manifold import TSNE
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from .util import DEFAULT_PLOTLY_COLORS, DEFAULT_GGPLOT_COLORS
from ..core import ExpMatrix
from ..latent import PCAModel

Numeric = Union[float, int]

_LOGGER = logging.getLogger(__name__)


def plot_cells(
        scores: pd.DataFrame,
        profile: pd.Series = None,
        colorscale: str = 'RdBu',
        reversescale: bool = True,
        emin: Numeric = None, emax: Numeric = None,
        cell_labels: pd.Series = None,
        cluster_order: List[str] = None,
        cluster_colors: Dict[str, str] = None,
        cluster_labels: Dict[str, str] = None,
        labelcells: bool = True,
        width: int = None, height: int = None,
        title: str = None,
        marker_size: int = 4, marker_color: str = None,
        opacity: Numeric = 0.7,
        xaxis_label: str = None, yaxis_label: str = None,
        showticklabels: bool = False,
        font_family: str = 'serif', font_size: Numeric = 32,
        automargin: bool = False,
        margin: Dict[str, int] = None,
        ticklen: int = 5,
        showscale: bool = None,
        colorbar_length: Numeric = 0.4, colorbar_label = None,
        colorbar_font_size: Numeric = None,
        showlegend: bool = False,
        showlabels: bool = True,
        labelsize: Numeric = 24,
        offsetlabels: bool = False,
        colorlabels: bool = False,
        showaxislabels: bool = True,
        default_colors: List[str] = None,
        colorscheme: str = 'plotly',
        plot_bgcolor: str = 'white',
        legend_title: str = None, legend_font_size: Numeric = None,
        labelopacity=1.0,
        legend_bgcolor=None,
        legend_xanchor: str = None, legend_x: Numeric = None,
        legend_yanchor: str = None, legend_y: Numeric = None) -> go.Figure:
    """Visualize cell similarities in a scatter plot."""

    valid_colorschemes = ['plotly', 'ggplot']

    if colorscheme not in valid_colorschemes:
        valid_str = ', '.join(['"%s"' % val for val in valid_colorschemes])
        raise ValueError(
            '`colorscheme` must be one of [%s], not "%s"!',
            valid_str, colorscheme)

    if default_colors is None:
        if colorscheme == 'plotly':
            default_colors = DEFAULT_PLOTLY_COLORS
        elif colorscheme == 'ggplot':
            default_colors = DEFAULT_GGPLOT_COLORS

    if margin is None:
        margin = {}

    if showscale is None:
        if profile is None:
            showscale = False
        else:
            showscale = True

    if width is None:
        if profile is None:
            width = 730
        else:
            if not showscale:
                width = 730
            elif colorbar_label is None:
                width = 800
            else:
                width = 820

    if height is None:
        height = 750

    cell_indices = pd.Series(
        index=scores.index, data=np.arange(scores.shape[0]))

    num_cells = scores.shape[0]

    if cell_labels is None:
        internal_cell_labels = pd.Series(
            index=scores.index, data=['Cells']*num_cells)
        if marker_color is None:
            marker_color = 'navy'
        cluster_colors = {'Cells': marker_color}
    else:
        internal_cell_labels = cell_labels

    vc = internal_cell_labels.value_counts()
    if cluster_order is None:
        cluster_order = vc.index.tolist()

    if cluster_colors is None:
        cluster_colors = {}

    for i, cluster in enumerate(cluster_order):
        if cluster in cluster_colors:
            continue
        #try:
        cluster_colors[cluster] = default_colors[i % len(default_colors)]
        #except IndexError:
        #    cluster_colors[cluster] = None

    if cluster_labels is None:
        cluster_labels = {}

    data = []
    for cluster in cluster_order:
        try:
            label = cluster_labels[cluster]
        except KeyError:
            label = cluster
        sel = (internal_cell_labels == cluster)

        if profile is not None:
            color=profile.loc[sel].values
        else:
            color=cluster_colors[cluster]

        sel_scores = scores.loc[sel]
        x = sel_scores.iloc[:, 0].values
        y = sel_scores.iloc[:, 1].values

        if labelcells:
            text = sel_scores.index.tolist()
            text = ['%s (%d)' % (t, cell_indices.loc[t]) for t in text]
        else:
            text = None

        colorbar = dict(
            len=colorbar_length,
            outlinewidth=1,
            outlinecolor='black',
            ticks='outside',
            ticklen=5,
            title=colorbar_label,
            titleside='right',
            titlefont=dict(size=colorbar_font_size),
            tickfont=dict(size=colorbar_font_size),
            separatethousands=True)

        trace = go.Scatter(
            x=x,
            y=y,
            text=text,
            name=label,
            mode='markers',
            marker=dict(
                size=marker_size, color=color,
                cmin=emin, cmax=emax,
                showscale=showscale,
                colorbar=colorbar,
                colorscale=colorscale,
                reversescale=reversescale,
                opacity=opacity))
        data.append(trace)

    if xaxis_label is None:
        xaxis_label = scores.columns[0]
    if yaxis_label is None:
        yaxis_label = scores.columns[1]

    if showticklabels:
        ticks = 'outside'
        default_margin = {
            'l': 140,
            'b': 100,
            't': 65,
            'r': 0,
        }
    else:
        ticks = None
        default_margin = {
            'l': 50,
            'b': 50,
            't': 115,
            'r': 90,
        }

    final_margin = default_margin
    final_margin.update(margin)

    legend = dict(
        title=legend_title,
        xanchor=legend_xanchor, yanchor=legend_yanchor,
        x=legend_x, y=legend_y,
        font=dict(size=legend_font_size),
        bgcolor=legend_bgcolor)

    annotations = []
    if cell_labels is not None and showlabels:

        for cluster in cluster_order:
            if cluster == 'Outliers' or cluster == 'Doublets':
                continue

            pos = scores.loc[internal_cell_labels == cluster].median(axis=0)
            xanchor = 'left'

            if offsetlabels:
                #ptp_x = np.ptp(scores.iloc[:, 0])
                #offset_x = ptp_x * 0.025
                #pos.iloc[0] = pos.iloc[0] + offset_x
                ptp_y = np.ptp(scores.iloc[:, 1])
                offset_y = ptp_y * 0.035
                pos.iloc[1] = pos.iloc[1] - offset_y

            try:
                text = cluster_labels[cluster]
            except KeyError:
                text = str(cluster)
                if text.startswith('Cluster '):
                    text = text[8:]

            text = ('<span style="text-shadow: '
                        'white -1px 0px 0px, '
                        'white 1px -0px 0px, '
                        'white 0px 1px 0px, '
                        'white 0px -1px 0px; '
                        'font-weight: bold">%s</span>') % text

            if colorlabels:
                font=dict(size=labelsize, color=cluster_colors[cluster])
            else:
                font=dict(size=labelsize, color='rgba(0,0,0,%s)' % str(labelopacity))

            ann = go.layout.Annotation(
                x=pos.iloc[0],
                y=pos.iloc[1],
                text=text,
                xanchor=xanchor,
                showarrow=False,
                font=font,
            )
            annotations.append(ann)

    # fix for plotly bug where xaxis label is misplaced if shoticklabels=False
    if not showticklabels:
        showticklabels = True
        tickfont = dict(color=plot_bgcolor)
    else:
        tickfont = dict()

    if not showaxislabels:
        titlefont = dict(color=plot_bgcolor)
    else:
        titlefont = dict()

    layout = go.Layout(
        width=width, height=height, margin=final_margin,
        font=dict(family=font_family, size=font_size),
        xaxis=dict(linecolor='black', title=xaxis_label, titlefont=titlefont,
                   automargin=automargin, showline=True, tickfont=tickfont,
                   ticks=ticks, ticklen=ticklen, showticklabels=showticklabels),
        yaxis=dict(linecolor='black', title=yaxis_label, titlefont=titlefont,
                   automargin=automargin, showline=True, tickfont=tickfont,
                   ticks=ticks, ticklen=ticklen, showticklabels=showticklabels),
        title=title,
        legend=legend,
        annotations=annotations,
        showlegend=showlegend,
        plot_bgcolor=plot_bgcolor)

    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_cells_random_order(
        scores: pd.DataFrame, cell_labels: pd.Series,
        cluster_order: List[str] = None,
        cluster_colors: Dict[str,str] = None,
        seed: int = 0,
        **kwargs) -> go.Figure:

    # determine cluster order
    vc = cell_labels.value_counts()
    if cluster_order is None:
        cluster_order = vc.index.tolist()    

    # determine cluster colors
    if cluster_colors is None:
        cluster_colors = {}

    for i, cluster in enumerate(cluster_order):
        if cluster not in cluster_colors:
            try:
                cluster_colors[cluster] = DEFAULT_PLOTLY_COLORS[i]
            except IndexError:
                cluster_colors[cluster] = None

    # generate regular figure (necessary to get a legend)
    regular_fig = plot_cells(
        scores, cell_labels=cell_labels,
        cluster_order=cluster_order,
        cluster_colors=cluster_colors, **kwargs)

    num_clusters = len(cluster_order)
    colorscale = []
    cluster_mapping = {}
    for i, cluster in enumerate(cluster_order):
        colorscale.append([i / (num_clusters-1), cluster_colors[cluster]])
        cluster_mapping[cluster] = i / (num_clusters-1)

    profile = cell_labels.map(cluster_mapping).astype(np.float64)

    # shuffle cells
    scores = scores.sample(axis=0, frac=1.0, random_state=seed, replace=False)
    profile = profile.loc[scores.index]

    # generate figure with random cell order
    # => no cell labels, instead use profile
    # => use custom colorscale, but hide colorbar
    kwargs['colorscale'] = colorscale
    kwargs['showscale'] = False

    random_order_fig = plot_cells(scores, profile=profile, **kwargs)

    return random_order_fig, regular_fig
