# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Tests for the `PCAModel` class."""

import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import pearsonr

from monet.core import ExpMatrix
from monet.latent import PCAModel
from monet.helper import assert_frame_not_equal


def test_pca_model(matrix, pc_scores):
    pca_model = PCAModel(num_components=1)
    test_scores = pca_model.fit_transform(matrix).iloc[:, 0]

    corr = pearsonr(test_scores, pc_scores.iloc[:, 0])[0]
    assert corr >= 0.99


def test_inverse_pca(pca_model, pc_scores, matrix, restored_matrix):    
    test_restored_matrix = pca_model.inverse_transform(pc_scores)
    assert_frame_equal(test_restored_matrix, restored_matrix,
                       check_exact=False, check_less_precise=True)
    assert_frame_not_equal(matrix, restored_matrix,
                       check_exact=False, check_less_precise=True)

