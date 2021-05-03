# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Tests for the `CompressedData` class."""

import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import pearsonr

from monet.core import ExpMatrix
from monet.latent import CompressedData


def test_from_matrix(matrix, compressed_data):
    compressed_data_test = CompressedData.from_matrix(matrix, num_components=1)
    corr = pearsonr(compressed_data_test.pc_scores.iloc[:, 0],
                    compressed_data.pc_scores.iloc[:, 0])[0]
    assert corr >= 0.99
