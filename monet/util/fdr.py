# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Utility functions for determining False Discovery Rate (FDR) thresholds."""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


def get_adjusted_pvalues(pvals: pd.Series, fdr_thresh: float = 0.05) \
        -> Tuple[pd.Series, float]:
    """
    Function that controls FDR rate.
    Accepts an unsorted list of p-values and an FDR threshold (1).
    Returns:
    1) the FDR associated with each p-value,
    2) the p-value cutoff for the given FDR.

    References:
    (1) Storey, J. D., & Tibshirani, R. (2003). Statistical significance
    for genomewide studies. Proceedings of the National Academy of Sciences,
    100(16), 9440-9445. https://doi.org/10.1073/pnas.1530509100
    """
    m = pvals.size

    # sort list of p-values
    sort_ids = np.argsort(pvals) # returns indices for sorting
    p_sorted = pvals.values[sort_ids] # sorts the list

    adj_p = np.nan * np.zeros(len(p_sorted), dtype=np.float64)
    crit_p = 0

    # go over all p-values, starting with the largest
    crossed = False
    adj_p[-1] = p_sorted[-1]
    i = m-2
    while i >= 0:
        FP = m*p_sorted[i]  # calculate false positives
        FDR = FP / (i+1)  # calculate FDR
        adj_p[i] = min(FDR, adj_p[i+1])
        if FDR <= fdr_thresh and not crossed:
            crit_p = p_sorted[i]
            crossed = True
        i -= 1

    # reverse sorting
    unsort_ids = np.argsort(sort_ids)  # indices for reversing the sort
    adj_p = adj_p[unsort_ids]
    adj_p = pd.Series(index=pvals.index, data=adj_p, name='adjusted_pval')

    return adj_p, crit_p


def fdr_bh(p_values,q):
	# implements Benjamini Hochberg procedure
	a = np.argsort(p_values)
	n = a.size
	k = 0

	for i in range(n):
		if p_values[a[i]] <= q*((i+1)/float(n)):
			k = i+1

	crit_p = 0.0
	if k > 0:
		crit_p = p_values[a[k-1]]

	return k,crit_p


def fdr_bh_general(p_values,q):
	# implements Benjamini Hochberg procedure that guarantees
	# FDR control under arbitrary test dependencies
	n = p_values.size

	q_adj = 0
	for i in range(n):
		q_adj += (1/float(i+1))
	q_adj = q/q_adj

	return fdr_bh(p_values,q_adj)
