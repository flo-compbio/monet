# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Tests for the `ENHANCE` command-line interface."""

import os

import pytest
from pandas.testing import assert_frame_equal
import numpy as np

from monet.core import ExpMatrix


def test_run(expression_tsv_file, tmpdir, denoised_matrix):
    output_file = tmpdir.join('denoised_expression.tsv').strpath

    os.system('enhance.py -f %s -o %s' % (expression_tsv_file, output_file))

    denoised_matrix_test = ExpMatrix.load_tsv(output_file).astype(np.float32)

    assert np.allclose(denoised_matrix_test, denoised_matrix, atol=1e-4)

    #assert_frame_equal(denoised_matrix_test, denoised_matrix,
    #                   check_exact=False, check_less_precise=True)
