# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

import click
import logging
import sys

import numpy as np

from ..core import ExpMatrix
from . import EnhanceModel

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.option('-f', '--input-file',
              help='Path to plain-text input file containing the raw UMI count '
                   'matrix.')
@click.option('-o', '--output-file',
              help='Path to plain-text output file for saving the denoised '
                   'expression matrix.')
@click.option('--sep', default='\t', show_default=False,
              help='Separator used in input file. The output file will '
                   'use this separator as well.  [default: \\t]')
@click.option('-d', '--num-components',
               help='The number of principal components to use.')
@click.option('-n', '--agg-max-frac-neighbors',
              default=0.01, show_default=True,
              help='Limit the number of neighbors to use in the aggregation '
              'step to the specified fraction of the total number of cells in '
              'the data.')
@click.option('--use-double-precision', is_flag=True,
                help='Use double-precision floating point format. '
                     '(Requires twice the amount of memory.)')
def run_enhance(input_file, output_file, sep, num_components,
                agg_max_frac_neighbors,
                use_double_precision):
    """Run ENHANCE."""

    # configure root logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    format_ = '[%(asctime)s] (%(name)s) %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_, date_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    matrix = ExpMatrix.load_tsv(input_file, sep=sep)

    denoising_model = EnhanceModel(
        num_components=num_components,
        agg_max_frac_neighbors=agg_max_frac_neighbors,
        use_double_precision=use_double_precision)
    denoised_matrix = denoising_model.fit_transform(matrix)

    denoised_matrix.save_tsv(output_file, sep=sep)
