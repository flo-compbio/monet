# Copyright (c) 2015-2017, 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `StaticGSEResult` and `RankBasedGSEResult` classes."""

from typing import Iterable, List
import logging
import hashlib

import numpy as np

from xlmhg import mHGResult
from . import GeneSet

_LOGGER = logging.getLogger(__name__)


class SimpleEnrichmentResult:
    """Result of a hypergeometric test for gene set enrichment.
    
    Parameters
    ----------
    gene_set : `genometools.basic.GeneSet`
        See :attr:`gene_set`.
    N : int
        See :attr:`N`.
    n : int
        See :attr:`n`.
    selected_genes : iterable of `ExpGene`
        See :attr:`selected_genes`.
    pval : float
        See :attr:`pval`.

    Attributes
    ----------
    gene_set : `genometools.basic.GeneSet`
        The gene set.
    N : int
        The total number of genes in the analysis.
    n : int
        The number of genes selected.
    selected_genes : set of `ExpGene`
        The genes from the gene set found present.
    pval : float
        The hypergeometric p-value.   
    """
    def __init__(
            self,
            gene_set: GeneSet,
            N: int,
            n: int,
            selected_genes: Iterable[str],
            pval: float):

        self.gene_set = gene_set
        self.N = N
        self.n = n
        self.selected_genes = set(selected_genes)
        self.pval = pval

    def __repr__(self):
        return '<%s object (pval=%.1e, gene_set_id="%s", hash="%s">' \
               % (self.__class__.__name__,
                  self.pval, self.gene_set._id, self.hash)

    def __str__(self):
        return '<%s object (pval=%.1e; N=%d; K=%d; n=%d; k=%d; gene_set=%s)>' \
               % (self.__class__.__name__, self.pval, self.N, self.K, self.n,
                  self.k, str(self.gene_set))

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return self.hash == other.hash
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def hash(self) -> str:
        data_str = ';'.join(
            [str(repr(var)) for var in
             [self.gene_set, self.N, self.n, sorted(self.selected_genes),
              self.pval]])
        data = data_str.encode('UTF-8')
        return str(hashlib.md5(data).hexdigest())

    @property
    def K(self) -> int:
        return self.gene_set.size

    @property
    def k(self) -> int:
        return len(self.selected_genes)

    @property
    def fold_enrichment(self) -> float:
        """Returns the fold enrichment of the gene set.

        Fold enrichment is defined as ratio between the observed and the
        expected number of gene set genes present.
        """
        expected = self.K * (self.n/float(self.N))
        return self.k / expected

    def get_pretty_format(self, max_name_length: int = 0) -> str:
        """Returns a nicely formatted string describing the result.

        Parameters
        ----------
        max_name_length: int [0]
            The maximum length of the gene set name (in characters). If the
            gene set name is longer than this number, it will be truncated and
            "..." will be appended to it, so that the final string exactly
            meets the length requirement. If 0 (default), no truncation is
            performed. If not 0, must be at least 3.

        Returns
        -------
        str
            The formatted string.

        Raises
        ------
        ValueError
            If an invalid length value is specified.
        """

        if max_name_length < 0 or (1 <= max_name_length <= 2):
            raise ValueError('max_name_length must be 0 or >= 3.')

        gs_name = self.gene_set._name
        if max_name_length > 0 and len(gs_name) > max_name_length:
            assert max_name_length >= 3
            gs_name = gs_name[:(max_name_length - 3)] + '...'

        param_str = '(%d/%d @ %d/%d, pval=%.1e, fe=%.1fx)' \
                % (self.k, self.K, self.n, self.N,
                   self.pval, self.fold_enrichment)

        return '%s %s' % (gs_name, param_str)


class RankBasedEnrichmentResult(mHGResult):
    """Result of an XL-mHG-based test for gene set enrichment.

    This class inherits from `xlmhg.mHGResult`.

    Parameters
    ----------
    gene_set : `genometools.basic.GeneSet`
        See :attr:`gene_set` attribute.
    N: int
        The total number of genes in the ranked list.
        See also :attr:`xlmhg.mHGResult.N`.
    indices: `numpy.ndarray` of integers
        The indices of the gene set genes in the ranked list.
    ind_genes: list of str
        See :attr:`ind_genes` attribute.
    X: int
        The XL-mHG X parameter.
    L: int
        The XL-mHG L parameter.
    stat: float
        The XL-mHG test statistic.
    cutoff: int
        The cutoff at which the XL-mHG test statistic was attained.
    pval: float
        The XL-mHG p-value.
    pval_thresh: float, optional
        The p-value threshold used in the analysis. [None]
    escore_pval_thresh: float, optional
        The hypergeometric p-value threshold used for calculating the E-score.
        If not specified, the XL-mHG p-value will be used, resulting in a
        conservative E-score. [None]
    escore_tol: float, optional
        The tolerance used for calculating the E-score. [None]

    Attributes
    ----------
    gene_set : `genometools.basic.GeneSet`
        The gene set.
    ind_genes : list of str
        The names of the genes corresponding to the indices.
    """
    def __init__(
            self,
            gene_set: GeneSet,
            N: int,
            indices: np.ndarray,
            ind_genes: Iterable[str],
            X: int, L: int,
            stat: float,
            cutoff: int,
            pval: float,
            pval_thresh: float = None,
            escore_pval_thresh: float = None,
            escore_tol: float = None):

        # call parent constructor
        mHGResult.__init__(self, N, indices, X, L, stat, cutoff, pval,
                           pval_thresh=pval_thresh,
                           escore_pval_thresh=escore_pval_thresh,
                           escore_tol=escore_tol)

        if len(ind_genes) != indices.size:
            raise ValueError(
                'The number of genes must match the number of indices.')

        self.gene_set = gene_set
        self.ind_genes = list(ind_genes)

    # @classmethod
    # def from_mHGResult(...)

    def __repr__(self):
        return '<%s object (N=%d, gene_set_id="%s", hash="%s">' \
               % (self.__class__.__name__,
                  self.N, self.gene_set._id, self.hash)

    def __str__(self):
        return '<%s object (gene_set=%s, pval=%.1e)>' \
               % (self.__class__.__name__, str(self.gene_set), self.pval)

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return self.hash == other.hash
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def hash(self) -> str:
        data_str = ';'.join(
            [super().hash] +
            [str(repr(var)) for var in [self.gene_set, self.ind_genes]])
        data = data_str.encode('UTF-8')
        return str(hashlib.md5(data).hexdigest())

    @property
    def genes_above_cutoff(self) -> List[str]:
        return self.ind_genes[:self.k]

    def get_pretty_format(self, omit_param=True, max_name_length=0):
        # TO-DO: clean up, commenting
        gs_name = self.gene_set._name
        if max_name_length > 0 and len(gs_name) > max_name_length:
            assert max_name_length >= 3
            gs_name = gs_name[:(max_name_length - 3)] + '...'
        gs_str = gs_name + ' (%d / %d @ %d)' % \
                           (self.k, len(self.ind_genes), self.cutoff)
        param_str = ''
        if not omit_param:
            param_str = ' [X=%d,L=%d,N=%d]' % (self.X, self.L, self.N)
        details = ', p=%.1e, e=%.1fx%s' % (self.pval, self.escore, param_str)
        return '%s%s' % (gs_str, details)
