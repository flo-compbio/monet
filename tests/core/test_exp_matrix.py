# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Tests for the `ExpMatrix` class."""

import pytest
import numpy as np

from monet.core import ExpMatrix


def test_load_npz(expression_npz_file):

    matrix = ExpMatrix.load_npz(expression_npz_file)
    assert matrix.values.dtype == np.uint32
    assert matrix.num_genes == 14384
    assert matrix.num_cells == 100


def test_save_npz(matrix, tmpdir):
    output_file = tmpdir.join('expression.npz').strpath
    matrix.save_npz(output_file)

    recovered_matrix = ExpMatrix.load_npz(output_file)

    assert recovered_matrix.equals(matrix)


def test_load_tsv(expression_tsv_file):

    matrix = ExpMatrix.load_tsv(expression_tsv_file)
    assert matrix.values.dtype == np.uint32
    assert matrix.num_genes == 14384
    assert matrix.num_cells == 100


def test_save_tsv(matrix, tmpdir):
    output_file = tmpdir.join('expression.tsv').strpath
    matrix.save_tsv(output_file)

    recovered_matrix = ExpMatrix.load_tsv(output_file)

    assert recovered_matrix.equals(matrix)
