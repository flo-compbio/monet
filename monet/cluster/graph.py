# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import Tuple

import pandas as pd

from .. import util
from scanpy import pp
from scanpy import tl

from ..core import ExpMatrix
from ..latent import PCAModel

_LOGGER = logging.getLogger(__name__)


def cluster_cells_leiden(
        matrix: ExpMatrix,
        num_components: int = 50,
        resolution: float = 0.8,
        pca_model: PCAModel = None) -> Tuple[pd.Series, PCAModel]:

    _LOGGER.info('Performing graph-based clustering using the Leiden '
                 'algorithm with resolution=%s', str(resolution))

    if pca_model is None:
        pca_model = PCAModel(num_components=num_components)
        pc_scores = pca_model.fit_transform(matrix)

    else:
        pc_scores = pca_model.transform(matrix)

    adata = ExpMatrix(pc_scores.T).to_anndata()
    adata.obsm['pc_scores'] = pc_scores.values

    # determine nearest-neighbors
    pp.neighbors(adata, use_rep='pc_scores')

    # perform graph-based clustering
    tl.leiden(adata, resolution=resolution)

    cell_labels = adata.obs['leiden'].astype(int)
    vc = cell_labels.value_counts()
    _LOGGER.info('Generated %d clusters.', vc.size)

    #cluster_mapping = dict([i, 'Cluster %d' % (i+1)] for i in range(vc.size))
    #cluster_mapping = dict([i, str(i)] for i in range(vc.size))
    #cell_labels = cell_labels.map(cluster_mapping)

    return cell_labels, pca_model
