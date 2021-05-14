# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import os
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np

from anndata import AnnData
from ..core import ExpMatrix
from ..visualize.util import DEFAULT_PLOTLY_COLORS, DEFAULT_GGPLOT_COLORS

_LOGGER = logging.getLogger(__name__)


class VelocytoData:
    """Class for storing Velocyto raw data."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(
            self,
            exp_matrix: ExpMatrix,
            spliced_matrix: ExpMatrix,
            unspliced_matrix: ExpMatrix):

        self.exp_matrix = exp_matrix

        # make sure data are aligned
        if not (exp_matrix.genes.identical(spliced_matrix.genes) and
                exp_matrix.genes.identical(unspliced_matrix.genes)):
            raise ValueError('Gene indices are not identical!')

        if not (exp_matrix.cells.identical(spliced_matrix.cells) and
                exp_matrix.cells.identical(unspliced_matrix.cells)):
            raise ValueError('Cell indices are not identical!')

        self.spliced_matrix = spliced_matrix
        self.unspliced_matrix = unspliced_matrix


    def to_anndata(self, cell_labels: pd.Series = None,
                   cluster_order: List[str] = None,
                   cluster_labels: Dict[str, str] = None,
                   cluster_colors: Dict[str, str] = None,
                   umap_scores: pd.DataFrame = None,
                   colorscheme: str = 'plotly',
                   default_colors = None) -> AnnData:

        if default_colors is None:
            if colorscheme == 'plotly':
                default_colors = DEFAULT_PLOTLY_COLORS
            elif colorscheme == 'ggplot':
                default_colors = DEFAULT_GGPLOT_COLORS

        if cell_labels is not None:
            combined = self.exp_matrix.cells.intersection(cell_labels.index)
        else:
            combined = self.exp_matrix.cells

        adata = self.exp_matrix.loc[:, combined].to_anndata()
        adata.layers['spliced'] = self.spliced_matrix.loc[:, combined].values.T
        adata.layers['unspliced'] = self.unspliced_matrix.loc[:, combined].values.T    

        if umap_scores is not None:
            umap_scores = umap_scores.loc[combined]
            adata.obsm['X_umap'] = umap_scores.values

        if cell_labels is None:
            # no cell labels to add, we're done
            #adata.uns['clusters_colors'] = None
            return adata

        if cluster_labels is None:
            cluster_labels = {}

        if cluster_colors is None:
            cluster_colors = {}

        # make sure cell labels are aligned
        cell_labels = cell_labels.loc[combined]

        # replace cell labels with provided cluster labels
        cell_labels = cell_labels.replace(cluster_labels)

        # determine value counts
        vc = cell_labels.value_counts()

        if cluster_order is not None:
            # replace cluster order entries with provided cluster labels
            cluster_order = [cluster_labels[c] if c in cluster_labels else c
                            for c in cluster_order]
            # add missing clusters, in order of # cells
            for cluster in vc.index:
                if cluster not in cluster_order:
                    cluster_order.append(cluster)

        else:
            # order clusters by number of cells
            cluster_order = cell_labels.value_counts().index.tolist()

        ### prepare cluster colors
        # first, replace cluster_colors keys with provided cluster labels
        cluster_colors = dict(
            [cluster_labels[k], v] if k in cluster_labels else [k, v]
            for k, v in cluster_colors.items())

        # then, use default colors for clusters with unspecified colors
        for i, cluster in enumerate(cluster_order):
            if cluster in cluster_colors:
                continue
            cluster_colors[cluster] = default_colors[i % len(default_colors)]

        # add plotly cluster colors
        adata.uns['plotly_cluster_colors'] = cluster_colors

        # add cluster information to anndata object
        # (cluster ordering is encoded in pd.Categorical object)
        clusters = pd.Categorical(cell_labels)    
        adata.obs['clusters'] = clusters
        adata.obs['clusters'].cat.reorder_categories(cluster_order, inplace=True)

        # add cluster color information
        colors = []
        for i, cluster in enumerate(cluster_order):
            col = cluster_colors[cluster]
            numbers = [int(s) for s in col[4:-1].split(', ')]
            x = ''.join(f'{n:02x}' for n in numbers)
            h = f'#{x}'
            colors.append(h)

        adata.uns['clusters_colors'] = colors

        return adata


    def save_npz(self, fpath: str, compressed: bool = True) -> None:
        """Save Velocyto data as a .npz file."""

        with open(os.path.expanduser(fpath), 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Saved Velocyto data to "%s".', fpath)

        fpath_expanded = os.path.expanduser(fpath)

        data = {}

        data['genes'] = np.array(self.exp_matrix.genes.tolist())
        data['cells'] = np.array(self.exp_matrix.cells.tolist())

        data['matrix'] = np.array(self.exp_matrix.values.T, copy=False)
        data['spliced_matrix'] = \
                np.array(self.spliced_matrix.values.T, copy=False)
        data['unspliced_matrix'] = \
                np.array(self.unspliced_matrix.values.T, copy=False)

        if compressed:
            np.savez_compressed(fpath_expanded, **data)
        else:
            np.savez(fpath_expanded, **data)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Saved Velocyto data with %d cells and %d genes -- '
            '.npz (Monet) format, %.1f MB (expression matrix hash: %s).',
            self.exp_matrix.num_cells, self.exp_matrix.num_genes,
            file_size_mb, self.exp_matrix.hash)


    @classmethod
    def load_npz(cls, fpath):
        """Load expression matrix from a .npz file."""

        fpath_expanded = os.path.expanduser(fpath)

        data = np.load(fpath_expanded, allow_pickle=True)
        genes = data['genes']
        cells = data['cells']

        exp_data = np.array(data['matrix'].T, order='F', copy=False)
        spliced_data = \
                np.array(data['spliced_matrix'].T, order='F', copy=False)
        unspliced_data = \
                np.array(data['unspliced_matrix'].T, order='F', copy=False)

        exp_matrix = ExpMatrix(exp_data, genes=genes, cells=cells)
        spliced_matrix = ExpMatrix(spliced_data, genes=genes, cells=cells)
        unspliced_matrix = ExpMatrix(unspliced_data, genes=genes, cells=cells)

        velocyto_data = cls(exp_matrix, spliced_matrix, unspliced_matrix)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            '.npz (Monet) format, %.1f MB (expression matrix hash: %s).',
            exp_matrix.num_cells, exp_matrix.num_genes,
            file_size_mb, exp_matrix.hash)

        return velocyto_data
