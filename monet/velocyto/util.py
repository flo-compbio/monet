# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from anndata import AnnData
from scipy.sparse import issparse
import numpy as np


def anndata_ft_transform(adata: AnnData) -> None:
    """Apply Freeman-Tukey transform to AnnData object."""

    if issparse(adata.X):
        X = adata.X.data
    else:
        X = adata.X

    Y = np.sqrt(X) + np.sqrt(X+1)

    if issparse(adata.X):
        adata.X.data[:] = Y
    else:
        adata.X = Y
