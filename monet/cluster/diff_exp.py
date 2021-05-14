# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import Iterable
from math import floor, ceil

import pandas as pd
import numpy as np
from scipy.stats import binom, mannwhitneyu
import plotly.graph_objs as go

from ..core import ExpMatrix
from .. import util

_LOGGER = logging.getLogger(__name__)


def get_diff_pvalues(
        ref_matrix: ExpMatrix, comp_matrix: ExpMatrix) -> pd.Series:

    ref_num_transcripts = ref_matrix.median_transcript_count
    comp_num_transcripts = comp_matrix.median_transcript_count
    num_transcripts = (ref_num_transcripts + comp_num_transcripts) / 2.0

    ref_matrix = ref_matrix.scale(num_transcripts)
    comp_matrix = comp_matrix.scale(num_transcripts)

    ref_num_cells = ref_matrix.num_cells
    comp_num_cells = comp_matrix.num_cells

    expressed = ((ref_matrix.sum(axis=1) + comp_matrix.sum(axis=1)) > 0)
    ref_matrix = ref_matrix.loc[expressed]
    comp_matrix = comp_matrix.loc[expressed]

    genes = ref_matrix.index.copy()
    num_genes = len(genes)
    pvals = np.ones(num_genes, dtype=np.float64)
    for i in range(num_genes):

        x1 = ref_matrix.iloc[i, :].values
        x2 = comp_matrix.iloc[i, :].values

        _, p = mannwhitneyu(x1, x2, alternative='two-sided')
        pvals[i] = p

    pvals[pvals == 0] = np.nextafter(0, 1)
    # convert to series
    pvals = pd.Series(index=genes, data=pvals, name='pval')
    pvals.index.name = 'gene'
    return pvals



def get_DE_genes(
        ref_matrix: ExpMatrix, ref_labels: pd.Series,
        comp_matrix: ExpMatrix, comp_labels: pd.Series,
        fdr_thresh: float = 0.05,
        min_fold_change: float = 1.2,
        min_exp: float = 0.01,
        ignore_outliers: bool = True,
        sel_clusters: Iterable[str] = None) -> pd.DataFrame:

    genes = ref_matrix.genes & comp_matrix.genes
    ref_matrix = ref_matrix.loc[genes]
    comp_matrix = comp_matrix.loc[genes]

    ref_vc = ref_labels.value_counts()
    comp_vc = comp_labels.value_counts()

    shared_clusters = ref_vc.index & comp_vc.index

    if sel_clusters is None:
        sel_clusters = shared_clusters.tolist()
    else:
        ignore_outliers = False

    cluster_data = {}

    for cluster in sel_clusters:
        if cluster == 'Outliers' and ignore_outliers:
            _LOGGER.info('Ignoring "Outliers" cluster...')
            continue
        _LOGGER.info('Now processing cluster "%s"...', cluster)

        ref_sel_matrix = ref_matrix.loc[:, ref_labels == cluster]
        comp_sel_matrix = comp_matrix.loc[:, comp_labels == cluster]
        _LOGGER.info('Number of cells in reference/comparison matrix: %d / %d',
                     ref_sel_matrix.shape[1], comp_sel_matrix.shape[1])

        pvals = get_diff_pvalues(ref_sel_matrix, comp_sel_matrix)

        qvals, crit_p = \
            util.get_adjusted_pvalues(pvals, fdr_thresh)
        _LOGGER.info('Number of significant tests: %d',
                     (qvals <= fdr_thresh).sum())

        df = qvals.to_frame()
        df = df.loc[df['adjusted_pval'] <= fdr_thresh]

        scaled_ref_sel_matrix = ref_sel_matrix.scale(10000)
        scaled_comp_sel_matrix = comp_sel_matrix.scale(10000)

        scaled_ref_sel_matrix.values[scaled_ref_sel_matrix.values < min_exp] = min_exp
        scaled_comp_sel_matrix.values[scaled_comp_sel_matrix.values < min_exp] = min_exp

        fold_change = scaled_comp_sel_matrix.loc[df.index].mean(axis=1) / \
            scaled_ref_sel_matrix.loc[df.index].mean(axis=1)

        sel = (fold_change > 1.0)
        df_up = df.loc[sel].copy()
        df_up['fold_change'] = fold_change
        df_up = df_up.loc[df_up['fold_change'] >= min_fold_change]
        df_up = df_up.sort_values(by='fold_change', ascending=False)

        df_down = df.loc[~sel].copy()
        df_down['fold_change'] = 1.0 / fold_change
        df_down = df_down.loc[df_down['fold_change'] >= min_fold_change]
        df_down = df_down.sort_values(by='fold_change', ascending=False)

        cluster_data['%s_up' % cluster] = df_up
        cluster_data['%s_down' % cluster] = df_down

    # create multi_index
    df = pd.concat(cluster_data.values(), keys=cluster_data.keys())

    return df


def get_diff_pvalues_poisson(
        ref_matrix: ExpMatrix, comp_matrix: ExpMatrix) -> pd.Series:

    genes = ref_matrix.genes & comp_matrix.genes

    ref_num_transcripts = ref_matrix.median_transcript_count
    comp_num_transcripts = comp_matrix.median_transcript_count
    num_transcripts = (ref_num_transcripts + comp_num_transcripts) / 2.0

    ref_matrix = ref_matrix.scale(num_transcripts)
    comp_matrix = comp_matrix.scale(num_transcripts)

    ref_matrix = ref_matrix.loc[genes]
    comp_matrix = comp_matrix.loc[genes]

    ref_num_cells = ref_matrix.num_cells
    comp_num_cells = comp_matrix.num_cells

    expressed = ((ref_matrix.sum(axis=1) + comp_matrix.sum(axis=1)) > 0)
    ref_matrix = ref_matrix.loc[expressed]
    comp_matrix = comp_matrix.loc[expressed]

    genes = ref_matrix.index.copy()
    num_genes = len(genes)
    pvals = np.ones(num_genes, dtype=np.float64)
    for i in range(num_genes):

        k1 = ref_matrix.iloc[i, :].sum()
        k2 = comp_matrix.iloc[i, :].sum()

        k = k1 + k2

        # make sure k is integer using ceil()
        k_ceil = int(ceil(k))

        # calculate factor and adjust k2
        f = k_ceil / k
        k2 *= f

        # make sure k1 is integer using floor()
        # this is results in slightly conservative p-values
        k2_floor = int(floor(k2))

        # calculate p of the binomal by taking n1 and n2 into account
        p = comp_num_cells / (ref_num_cells + comp_num_cells)

        # what is the probability of getting k or greater (out of n) randomly?        
        # calculate lower tail: 
        pvals[i] = binom.sf(k2_floor-1, k_ceil, p)

    pvals[pvals == 0] = np.nextafter(0, 1)
    # convert to series
    pvals = pd.Series(index=genes, data=pvals, name='pval')
    pvals.index.name = 'gene'
    return pvals


def get_DE_genes_poisson(
        ref_matrix: ExpMatrix, ref_labels: pd.Series,
        comp_matrix: ExpMatrix, comp_labels: pd.Series,
        fdr_thresh: float = 0.05,
        min_exp: float = 0.01,
        min_fold_change: float = 1.2,
        ignore_outliers: bool = True,
        sel_clusters: Iterable[str] = None) -> pd.DataFrame:

    ref_vc = ref_labels.value_counts()
    comp_vc = comp_labels.value_counts()

    shared_clusters = ref_vc.index & comp_vc.index

    if sel_clusters is None:
        sel_clusters = shared_clusters.tolist()
    else:
        ignore_outliers = False

    cluster_data = {}

    for cluster in sel_clusters:
        if cluster == 'Outliers' and ignore_outliers:
            _LOGGER.info('Ignoring "Outliers" cluster...')
            continue
        _LOGGER.info('Now processing cluster "%s"...', cluster)

        ref_sel_matrix = ref_matrix.loc[:, ref_labels == cluster]
        comp_sel_matrix = comp_matrix.loc[:, comp_labels == cluster]

        # first, test for up-regulation
        up_pvals = get_diff_pvalues_poisson(ref_sel_matrix, comp_sel_matrix)
        # second, test for down-regulation
        down_pvals = get_diff_pvalues_poisson(comp_sel_matrix, ref_sel_matrix)

        up_pvals = up_pvals.reset_index()
        up_pvals['direction'] = 'up'

        down_pvals = down_pvals.reset_index()
        down_pvals['direction'] = 'down'

        df = pd.concat([up_pvals, down_pvals], axis=0, ignore_index=True)
        #print(df)

        qvals, qval_thresh = \
            util.get_adjusted_pvalues(df['pval'], fdr_thresh)
        _LOGGER.info('Number of significant tests: %d',
                     (qvals <= fdr_thresh).sum())

        df['adjusted_pval'] = qvals
        del df['pval']

        df = df.loc[df['adjusted_pval'] <= fdr_thresh]
        print('test:', df.shape)

        scaled_ref_sel_matrix = ref_sel_matrix.scale(10000)
        scaled_comp_sel_matrix = comp_sel_matrix.scale(10000)

        scaled_ref_sel_matrix.values[scaled_ref_sel_matrix.values < min_exp] = min_exp
        scaled_comp_sel_matrix.values[scaled_comp_sel_matrix.values < min_exp] = min_exp

        sel = (df['direction'] == 'up')

        df_up = df.loc[sel].copy()
        df_up.set_index('gene', inplace=True)

        ref_mean = scaled_ref_sel_matrix.loc[df_up.index].mean(axis=1)
        comp_mean = scaled_comp_sel_matrix.loc[df_up.index].mean(axis=1)
        df_up['fold_change'] = comp_mean / ref_mean

        df_up = df_up.loc[df_up['fold_change'] >= min_fold_change]
        df_up = df_up.sort_values(by='fold_change', ascending=False)

        df_down = df.loc[~sel].copy()
        df_down.set_index('gene', inplace=True)

        ref_mean = scaled_ref_sel_matrix.loc[df_down.index].mean(axis=1)
        comp_mean = scaled_comp_sel_matrix.loc[df_down.index].mean(axis=1)
        df_down['fold_change'] = ref_mean / comp_mean

        df_down = df_down.loc[df_down['fold_change'] >= min_fold_change]
        df_down = df_down.sort_values(by='fold_change', ascending=False)

        cluster_data['%s_up' % cluster] = df_up
        cluster_data['%s_down' % cluster] = df_down

    # create multi_index
    df = pd.concat(cluster_data.values(), keys=cluster_data.keys())

    return df


def plot_diff_exp(df: pd.DataFrame, cluster: str) -> go.Figure:
    
    df_up = df.loc[cluster+'_up'].copy()
    df_down = df.loc[cluster+'_down'].copy()
    
    df_down['fold_change'] = 1 / df_down['fold_change']
    
    df_sel = pd.concat([df_up, df_down], axis=0)
    
    x = np.log2(df_sel['fold_change'])
    y = -np.log10(df_sel['adjusted_pval'])
    trace = go.Scatter(
        x=x,
        y=y,
        text=df_sel.index,
        mode='markers',
        marker=dict(size=5),
    )
    data = [trace]
    layout = go.Layout(
        font=dict(family='serif', size=32),
        width=700,
        height=600,
        plot_bgcolor='white',
        xaxis=dict(linecolor='black', ticks='outside', title='log<sub>2</sub>-FC'),
        yaxis=dict(linecolor='black', ticks='outside', title='-log<sub>10</sub>adj. p-value'),
        title=cluster,
    )
    fig = go.Figure(data=data, layout=layout)
    return fig
