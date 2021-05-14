# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Tuple, Iterable
import gc
import logging

import pandas as pd
import plotly.graph_objs as go

from ..core import ExpMatrix
from . import plot_ribo_mito_fractions
from .. import util

_LOGGER = logging.getLogger(__name__)


def preprocess_data(
        matrix: ExpMatrix, gene_table: pd.DataFrame, species='human',
        gene_min_cells_expressed: int = 3,
        max_mito_frac: float = 0.15, min_transcripts: int = 2000,
        cell_umi_thresh: int = 1000,
        sel_cells: Iterable[str] = None, sel_genes: Iterable[str] = None,
        title: str = None,
        seed: int = 0) \
            -> Tuple[ExpMatrix, go.Figure]:

    #genome_annotation_file = '~/data/ensembl/release_101/Homo_sapiens.GRCh38.101.gtf.gz'
    mito_genes = util.get_mitochondrial_genes(species=species)
    ribo_genes = util.get_ribosomal_genes(species=species)

    if sel_cells is None:
        # keep only "real" cells
        num_transcripts = matrix.sum(axis=0)
        num_total = matrix.num_cells
        matrix = matrix.loc[:, num_transcripts >= cell_umi_thresh]
        num_kept = matrix.num_cells
        _LOGGER.info(
            'Kept %d / %d barcodes with at least %d UMIs ("cells").',
            num_kept, num_total, cell_umi_thresh)
    else:
        matrix = matrix.loc[:, sel_cells]
        num_kept = matrix.num_cells

    num_total_genes = matrix.num_genes
    if sel_genes is None:
        # keep only expressed genes
        num_cells_expressed = (matrix > 0).sum(axis=1)
        matrix = matrix.loc[num_cells_expressed >= gene_min_cells_expressed]
        num_exp_genes = matrix.num_genes
        num_unexp_genes = num_total_genes - num_exp_genes
        _LOGGER.info('Removed %d unexpressed genes (%.1f %%).',
                    num_unexp_genes, 100*(num_unexp_genes/num_total_genes))
    else:
        num_exp_genes = num_total_genes

    # convert from Ensembl IDs to gene names
    # (keep only genes with known Ensembl IDs)
    gene_conv = gene_table.iloc[:, 0]
    is_known = matrix.genes.isin(gene_conv.index)

    matrix = matrix.loc[is_known]
    matrix.index = matrix.index.map(gene_conv)
    num_known_genes = matrix.num_genes
    num_unknown_genes = num_exp_genes - num_known_genes
    _LOGGER.info('Removed %d genes with unknown Ensembl IDs (%.1f %%).',
                 num_unknown_genes, 100*(num_unknown_genes/num_exp_genes))

    # consolidate memory
    gc.collect()
    matrix = matrix.copy()
    gc.collect()

    # collapse duplicate gene names
    matrix = matrix.groupby(matrix.genes).sum()
    num_unique_genes = matrix.num_genes
    _LOGGER.info('Collapsed %d duplicate genes.',
                 num_known_genes - num_unique_genes)

    # generate qc plot
    fig = plot_ribo_mito_fractions(matrix, species=species, title=title)

    # calculate fraction of mitochondrial transcripts
    # and apply mitochondrial fraction filter
    num_total_cells = matrix.num_cells
    num_transcripts = matrix.sum(axis=0)
    frac_mito = matrix.reindex(mito_genes).fillna(0).sum() / num_transcripts
    frac_ribo = matrix.reindex(ribo_genes).fillna(0).sum() / num_transcripts
    if sel_cells is None:
        matrix = matrix.loc[:, frac_mito <= max_mito_frac]
        num_mito_pass_cells = matrix.num_cells
        num_mito_fail_cells = num_total_cells - num_mito_pass_cells
        frac_mito_fail_cells = num_mito_fail_cells / num_total_cells
        _LOGGER.info(
            'Removed %d cells (%.1f %%) with > %.1f %% mitochondrial transcripts.',
            num_mito_fail_cells,
            100*frac_mito_fail_cells,
            100*max_mito_frac)
    else:
        frac_mito_fail_cells = None

    if sel_genes is None:
        # remove mitochondrial genes
        matrix = matrix.loc[~matrix.genes.isin(mito_genes)]
    else:
        matrix = matrix.loc[sel_genes]

    if sel_cells is None:
        # apply UMI filter
        num_transcripts = matrix.sum(axis=0)
        matrix = matrix.loc[:, num_transcripts >= min_transcripts]
        num_umi_pass_cells = matrix.num_cells
        num_umi_fail_cells = num_mito_pass_cells - num_umi_pass_cells
        _LOGGER.info(
            'Removed %d / %d cells (%.1f %%) with < %d '
            'non-mitochondrial transcripts ',
            num_umi_fail_cells, num_mito_pass_cells,
            100*(num_umi_fail_cells/num_mito_pass_cells),
            min_transcripts)

    # consolidate memory
    gc.collect()

    # shuffle cells
    matrix = matrix.sample(axis=1, frac=1.0, replace=False, random_state=seed)
    gc.collect()

    data = {
        'num_transcripts': num_transcripts.loc[matrix.cells],
        'frac_mito': frac_mito.loc[matrix.cells],
        'frac_ribo': frac_ribo.loc[matrix.cells],
    }
    cell_qc = pd.concat(data, axis=1)

    qc_data = {
        'raw_num_cells': num_kept,
        'frac_mito_fail_cells': frac_mito_fail_cells,
    }

    return matrix, cell_qc, qc_data, fig
