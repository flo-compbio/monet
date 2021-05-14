# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2019 Florian Wagner
#
# This file is part of Monet.

from pkg_resources import resource_filename

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from monet.core import ExpMatrix
from monet.latent import PCAModel
from monet.denoise import ENHANCE

@pytest.fixture(scope='session')
def expression_npz_file():
    return resource_filename(
            'monet', 'data/test/pbmc_100_expression.npz')

@pytest.fixture(scope='session')
def expression_tsv_file():
    return resource_filename(
            'monet', 'data/test/pbmc_100_expression.tsv')

@pytest.fixture(scope='session')
def matrix(expression_npz_file):
    matrix = ExpMatrix.load_npz(expression_npz_file)
    return matrix

@pytest.fixture(scope='session')
def pc_scores():
    score_file = resource_filename(
        'monet', 'data/test/pbmc_100_pc1_scores.tsv')
    pc_scores = pd.read_csv(score_file, sep='\t', index_col=0, header=0)
    return pc_scores

@pytest.fixture(scope='session')
def pca_model():
    pca_model_file = resource_filename(
        'monet', 'data/test/pbmc_100_pca_model.pickle')
    pca_model = PCAModel.load_pickle(pca_model_file)
    return pca_model

@pytest.fixture(scope='session')
def pbmc_1k_matrix():
    expression_file = resource_filename(
        'monet', 'data/test/pbmc_1k_expression.npz')
    return ExpMatrix.load_npz(expression_file)

@pytest.fixture(scope='session')
def pbmc_1k_denoising_model():
    denoising_model_file = resource_filename(
        'monet', 'data/test/pbmc_1k_denoising_model.npz')
    return ENHANCE.load_pickle(denoising_model_file)
