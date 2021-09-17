# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Tuple

import pandas as pd

import scanpy.tl as tl
import scanpy.pp as pp
import plotly.graph_objs as go

from ..core import ExpMatrix
from ..latent import PCAModel
from .cells import plot_cells


def force_plot(
        matrix: ExpMatrix,
        num_components: int = 50,
        transform_name: str = 'freeman-tukey',
        pca_model: PCAModel = None,
        **kwargs) -> Tuple[go.Figure, pd.DataFrame]:

    if pca_model is None:
        pca_model = PCAModel(num_components=num_components, transform_name=transform_name)
        pc_scores = pca_model.fit_transform(matrix)

    else:
        pc_scores = pca_model.transform(matrix)

    adata = ExpMatrix(pc_scores.T).to_anndata()
    adata.obsm['pc_scores'] = pc_scores.values

    # determine nearest-neighbors
    pp.neighbors(adata, use_rep='pc_scores')

    tl.draw_graph(adata)

    Y = adata.obsm['X_draw_graph_fa']

    scores = pd.DataFrame(
        index=adata.obs_names, columns=['Dim. 1', 'Dim. 2'], data=Y)

    fig = plot_cells(scores, **kwargs)

    return fig, scores
