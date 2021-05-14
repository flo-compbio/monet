# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
import time
from typing import Tuple, List
from math import sqrt, ceil

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


def cluster_cells_dbscan(
        tsne_scores: pd.DataFrame,
        min_cells_frac: float = 0.01,
        eps_frac: float = 0.03) -> Tuple[pd.Series, List[str]]:
    """Cluster cells by applying DBSCAN to t-SNE result (Galapagos)."""

    # prepare DBSCAN parameters
    ptp = np.ptp(tsne_scores.values, axis=0)
    eps = eps_frac * sqrt(np.sum(np.power(ptp, 2.0)))
    minPts = int(ceil(min_cells_frac * tsne_scores.shape[0]))
    _LOGGER.info('Performing DBSCAN with minPts=%d and eps=%.2f.',
                 minPts, eps)

    # perform DBSCAN
    dbscan_model = DBSCAN(algorithm='brute', min_samples=minPts, eps=eps)
    t0 = time.time()
    z = dbscan_model.fit_predict(tsne_scores.values)
    t1 = time.time()
    _LOGGER.info('Clustering with DBSCAN took %.1f s.' % (t1-t0))    
    cell_labels = pd.Series(index=tsne_scores.index, data=z)

    # sort clusters by size
    vc = cell_labels.value_counts()
    clusters = [c for c in vc.index if c != -1]
    cluster_labels = dict([
        (id_, 'Cluster %d' % (i+1)) for i, id_ in enumerate(clusters)])
    if -1 in vc.index:
        cluster_labels[-1] = 'Outliers'
    cell_labels = cell_labels.map(cluster_labels)
    clusters = list(cluster_labels.values())

    return cell_labels, clusters
