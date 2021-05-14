# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

"""Functions for plotting QC statistics."""

import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..util import get_ribosomal_genes, get_mitochondrial_genes
from ..visualize.util import DEFAULT_PLOTLY_COLORS


def plot_ribo_mito_fractions(
        matrix: ExpMatrix, species: str = 'human', title: str = None,
        num_cells: int = 1000, seed: int = 0, opacity: int = 0.7):
    """Plot fractions of ribosomal and mitochondrial transcripts per cell."""

    # Layout:
    #   t1 (ribo bar plot), t2 (ribo scatter plot)
    #   t3 (mito bar plot), t4 (mito scatter plot)

    if num_cells < matrix.num_cells:
        # randomly sample a set of cells
        np.random.seed(seed)
        sel = np.random.choice(matrix.num_cells, size=num_cells, replace=False)
        matrix = matrix.iloc[:, sel]

    def scatter_trace(matrix, sel_genes, color, opacity):
        num_umi = matrix.sum(axis=0)
        subset_index = matrix.genes.intersection(sel_genes)
        subset_sum = matrix.loc[subset_index].sum(axis=0)
        subset_frac = subset_sum / num_umi
        trace = go.Scatter(
            x=num_umi,
            y=100*subset_frac,
            text=matrix.cells,
            mode='markers',
            marker=dict(color=color, opacity=opacity))
        return trace


    def hist_trace(matrix, sel_genes, color, opacity):
        subset_index = matrix.genes.intersection(sel_genes)
        subset_sum = matrix.loc[subset_index].sum(axis=0)
        subset_frac = subset_sum / matrix.sum(axis=0)
        fillcol = 'rgba(%s, %f)' % (color[4:-1], opacity)
        trace = go.Histogram(
            y=100*subset_frac,
            histnorm='percent',
            autobiny=False,
            ybins=dict(start=0, end=100.01, size=5.00001),
            marker=dict(
                color=fillcol,
                line=dict(width=1.0, color=color)))
        return trace

    ribo_genes = get_ribosomal_genes(species)
    mito_genes = get_mitochondrial_genes(species)

    t1 = hist_trace(matrix, ribo_genes, DEFAULT_PLOTLY_COLORS[0], opacity)
    t2 = scatter_trace(matrix, ribo_genes, DEFAULT_PLOTLY_COLORS[0], opacity)
    t3 = hist_trace(matrix, mito_genes, DEFAULT_PLOTLY_COLORS[1], opacity)
    t4 = scatter_trace(matrix, mito_genes, DEFAULT_PLOTLY_COLORS[1], opacity)

    t1.xaxis = 'x'
    t1.yaxis = 'y'
    t2.xaxis = 'x2'
    t2.yaxis = 'y'

    t3.xaxis = 'x'
    t3.yaxis = 'y2'
    t4.xaxis = 'x2'
    t4.yaxis = 'y2'

    data = [t1, t2, t3, t4]

    l = 0.12
    b = 0.10
    ygap = 0.08
    xgap = 0.08
    y = (1.0-b-ygap)/2
    x = (1.0-l-xgap)/2
    num_transcripts = np.log10(matrix.sum(axis=0))
    ptp = np.ptp(num_transcripts)
    xmin = max(num_transcripts.min() - 0.025*ptp, 0)
    xmax = num_transcripts.max() + 0.025*ptp

    layout = go.Layout(
        title=title,
        width=1100,
        height=900,
        font=dict(family='serif', size=24),
        showlegend=False,
        plot_bgcolor='white',
        grid=dict(
            subplots=[['xy', 'x2y'], ['xy2', 'x2y2']]
        ),
        xaxis=dict(
            domain=[l, l+x],
            ticks='outside',
            range=[0, 100],
            linecolor='black',
            gridcolor='lightgray',
            title='Fraction of cells (%)',
        ),
        yaxis=dict(
            domain=[b+ygap+y, 1.0],
            ticks='outside',
            range=[0, 100],
            linecolor='black',
            gridcolor='lightgray',
            title='Fraction of ribos.<br> transcripts (%)',
        ),
        xaxis2=dict(
            domain=[l+x+xgap, 1],
            ticks='outside',
            range=[xmin, xmax],
            type='log',
            #dtick='D3',
            linecolor='black',
            gridcolor='lightgray',
            title='Total # of transcripts',
        ),
        yaxis2=dict(
            domain=[b, b+y],
            ticks='outside',
            range=[0, 100],
            linecolor='black',
            gridcolor='lightgray',
            title='Fraction of mitoch.<br> transcripts (%)',
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    return fig
