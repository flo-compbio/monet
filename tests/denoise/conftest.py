# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from pkg_resources import resource_filename

import pytest

from monet.core import ExpMatrix
from monet.latent import CompressedData

@pytest.fixture(scope='session')
def compressed_data():
    compressed_data_file = resource_filename(
        'monet', 'data/test/pbmc_100_compressed_data.pickle')
    return CompressedData.load_pickle(compressed_data_file)

@pytest.fixture(scope='session')
def restored_matrix():
    expression_file = resource_filename(
        'monet', 'data/test/pbmc_100_pc1_restored_expression.npz')
    return ExpMatrix.load_npz(expression_file)

@pytest.fixture(scope='session')
def decompressed_matrix():
    expression_file = resource_filename(
        'monet', 'data/test/pbmc_100_decompressed_expression.npz')
    return ExpMatrix.load_npz(expression_file)

@pytest.fixture(scope='session')
def denoised_matrix():
    expression_file = resource_filename(
        'monet', 'data/test/pbmc_100_denoised_expression.npz')
    return ExpMatrix.load_npz(expression_file)
