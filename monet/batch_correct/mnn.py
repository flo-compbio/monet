# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import Tuple, Union
import sys
import time

from ..core import ExpMatrix
from ..latent import PCAModel

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


def correct_mnn(
        pca_model: PCAModel,
        ref_matrix: ExpMatrix, target_matrix: ExpMatrix,
        k: int = 20, num_mnn: int = 5) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform batch correction in latent space using mutual nearest neighbors.

    This function implements a method very similar to the one proposed
    by Haghverdi et al. (Nat Biotech, 2018), except all operations are
    performed after projecting the data into a latent space defined by a PCA
    model.
    => PMID: 29608177
    => DOI: 10.1038/nbt.4091.

    Returns:
    ========
    1. The batch-corrected PC scores for the target matrix.
    2. The PC scores obtained by applying the PCA model to the
       reference matrix.
    """

    t0 = time.time()

    ### determine all MNN pairs
    _LOGGER.info('Determining all MNN pairs...')

    # transform data using PCA model
    ref_pc_scores = pca_model.transform(ref_matrix)
    target_pc_scores = pca_model.transform(target_matrix)
    
    # find k nearest neighbors in reference matrix
    # for all points in target matrix
    ref_nn_model = NearestNeighbors(algorithm='kd_tree', n_neighbors=k)
    ref_nn_model.fit(ref_pc_scores.values)
    neigh_ind = ref_nn_model.kneighbors(
        target_pc_scores.values, return_distance=False)

    # test if each point r in ref_matrix is a MNN
    # of point t in target matrix
    target_nn_model = NearestNeighbors(algorithm='kd_tree', n_neighbors=k)
    target_nn_model.fit(target_pc_scores.values)

    mnn = []
    mnn_indices = {}
    
    cur_neighbor = 0
    cur_idx = 0
    for t in range(neigh_ind.shape[0]):
        
        # use pre-determined nearest neighbors in reference
        ref_neighbors = neigh_ind[t, :]
        indices = []

        # get an adjacency matrix for the ref_neighbors in the target
        A = target_nn_model.kneighbors_graph(
                ref_pc_scores.iloc[ref_neighbors])
        for i in range(A.shape[0]):
            # each i corresponds to one point r in ref_neighbors
            if A[i, t] == 1:
                indices.append(cur_idx)
                mnn.append((t, ref_neighbors[i]))
                cur_idx += 1
                
        if indices:
            mnn_indices[cur_neighbor] = np.uint32(indices)
            cur_neighbor += 1


    # mnn is a 2-by-x array containing all MNN pairs
    # column 1 = point t index, column 2 = point r index
    mnn = np.uint32(mnn)

    # calculate correction vectors for all MNN pairs
    _LOGGER.info('Calculating batch correction vectors for all MNN pairs...')
    corr_vectors = np.empty((mnn.shape[0], pca_model.num_components_),
                            dtype=np.float32)
    for i in range(mnn.shape[0]):
        cv = ref_pc_scores.iloc[mnn[i, 1]].values - target_pc_scores.iloc[mnn[i, 0]].values
        corr_vectors[i, :] = cv
    
    # now we know which points in T have mutual neareast neighbors (=T_mut)
    # next, we will use those for batch correction
    #mnn_ind = np.unique(mnn[:, 0])
    T_mut = np.unique(mnn[:, 0])

    # apply the correction
    _LOGGER.info('Applying batch correction to target PC scores...'); sys.stdout.flush()
    
    # first, create NN model for T_mut
    mnn_nn_model = NearestNeighbors(algorithm='kd_tree', n_neighbors=num_mnn)
    mnn_nn_model.fit(target_pc_scores.iloc[T_mut])
    
    # then, find the closest MNNs for all points in target matrix
    # and use the mean of their batch correction vectors
    C = target_pc_scores.values.copy()
    nearest_mnn = mnn_nn_model.kneighbors(C, return_distance=False)
    for t in range(C.shape[0]):

        # determine the indices of all batch correction vectors
        # for point t
        pair_indices = []
        for i in nearest_mnn[t, :]:
            pair_indices.extend(mnn_indices[i])
        pair_indices = np.uint32(pair_indices)
        
        corr = corr_vectors[pair_indices, :].mean(axis=0)
        C[t, :] = C[t, :] + corr
    
    corrected_target_pc_scores = pd.DataFrame(
        index=target_pc_scores.index,
        columns=target_pc_scores.columns,
        data=C)
    
    t1 = time.time()
    _LOGGER.info('Batch correction using mutual nearest neighbors '
                 'took %.1f s.', t1-t0)

    return corrected_target_pc_scores, ref_pc_scores
