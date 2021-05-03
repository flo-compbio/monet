# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from ..core import ExpMatrix
from ..latent import PCAModel


def transfer_knn(
        pca_model: PCAModel, ref_matrix: ExpMatrix, ref_cell_labels: pd.Series,
        target_matrix: ExpMatrix,
        num_neighbors: int = 20, **kwargs) -> pd.Series:

    le = LabelEncoder()

    ref_pc_scores = pca_model.transform(ref_matrix)
    y_ref = le.fit_transform(ref_cell_labels.values)

    clf = KNeighborsClassifier(n_neighbors=num_neighbors, **kwargs)
    clf.fit(ref_pc_scores.values, y_ref)

    target_pc_scores = pca_model.transform(target_matrix)
    y_pred = clf.predict(target_pc_scores.values)    
    y_pred = le.inverse_transform(y_pred)

    target_cell_labels = pd.Series(index=target_matrix.cells, data=y_pred)

    return target_cell_labels
