# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Tests for the `ENHANCE` class."""

import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import pearsonr
import numpy as np

from monet.core import ExpMatrix
from monet.denoise import ENHANCE
from monet.denoise.util import aggregate_neighbors


@pytest.mark.skip(reason="Not used at the moment")
def test_agg_determine_num_components(matrix):
    """Test if determining the number of PCs using MCV works."""
    denoising_model = ENHANCE(agg_num_neighbors=1)
    denoising_model._determine_num_components(matrix)

    assert denoising_model.pca_num_components_ == 6


def test_enhance(matrix, denoised_matrix):
    """Don't do any aggregation, only focus on MCV."""
    denoising_model = ENHANCE()
    denoised_matrix_test = denoising_model.fit_transform(matrix)

    assert denoising_model.pca_num_components_ == 6
    #assert_frame_equal(denoised_matrix_test, denoised_matrix,
    #                   check_exact=False, check_less_precise=True)
    assert np.allclose(denoised_matrix_test, denoised_matrix)
