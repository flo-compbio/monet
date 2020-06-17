
# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from typing import Dict, Tuple
from ..core import ExpMatrix

import pandas as pd


def combine_matrices(datasets: Dict[str, ExpMatrix]) \
        -> Tuple[ExpMatrix, pd.Series]:
    """Combine multiple expression matrices (samples)."""

    # prepand dataset name to cell names for each dataset
    for name, matrix in datasets.items():
        matrix.cells = matrix.cells.to_series().apply(
            lambda x: '%s_%s' % (name, x)).values

    combined_matrix = pd.concat(datasets.values(), axis=1).fillna(0)
    
    cell_labels = combined_matrix.cells.to_series().str.split('_').apply(
        lambda x: x[0])
    
    return combined_matrix, cell_labels
