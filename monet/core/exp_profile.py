# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `ExpProfile` class."""

import importlib
import logging

import pandas as pd
import numpy as np

exp_matrix = importlib.import_module('.exp_matrix', package='monet.core')

_LOGGER = logging.getLogger(__name__)


class ExpProfile(pd.Series):
    """A gene expression profile.

    Inherits from `pd.Series`."""

    def __init__(self, *args, genes=None,  **kwargs):
        
        if genes is not None:
            if len(args) >= 2 or 'index' in kwargs:
                raise ValueError(
                    'Providing both `genes` and `index` is redundant!')
            kwargs['index'] = genes

        pd.Series.__init__(self, *args, **kwargs)

        if self.index.name is None:
            self.index.name = 'Genes'


    def __repr__(self):
        return '<%s instance with %d genes>' \
               % (self.__class__.__name__, self.num_genes)


    @property
    def _constructor(self):
        return ExpProfile

    @property
    def _constructor_expanddim(self):
        return exp_matrix.ExpMatrix

    @property
    def hash(self) -> str:
        from ..util import calculate_hash
        return calculate_hash(self.to_frame())

    @property
    def num_genes(self):
        return self.size

    @property
    def genes(self):
        """Alias for `Series.index`."""
        return self.index

    @genes.setter
    def genes(self, genes):
        self.index = genes

    @property
    def x(self):
        """Alias for `Series.values`."""
        return self.values


    @classmethod
    def load_tsv(cls, fpath, sep='\t', encoding='utf-8', 
                 index_col=0, header=0, squeeze=True, **kwargs):
        """Load expression profile from a text file.
        
        Wrapper around `pandas.read_csv()`."""

        profile = pd.read_csv(
            fpath, encoding=encoding, sep=sep,
            index_col=index_col, header=header,
            squeeze=True, **kwargs)

        profile = cls(profile)

        _LOGGER.info(
            'Loaded expression profile with %d genes -- '
            'plain-text format, (hash: %s).',
            profile.num_genes, profile.hash)

        return profile


    def save_tsv(self, fpath, sep='\t', encoding='utf-8',
                 float_format='%.5f', **kwargs):
        """Save expression profile as a plain-text file.

        Wrapper around `pandas.Series.to_csv()`.
        """

        self.to_csv(fpath, sep=sep, encoding=encoding, 
                    float_format=float_format, **kwargs)

        _LOGGER.info(
            'Saved expression profile with %d genes -- '
            'plain-text format (hash: %s).',
            self.num_genes, self.hash)
