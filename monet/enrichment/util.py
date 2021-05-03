# Copyright (c) 2015, 2016, 2020 Florian Wagner

# This file is part of Monet.

"""Utility functions for working with GO annotations."""

import sys
import os
import re
import hashlib
import logging
from collections import OrderedDict
from typing import Iterable, List

import pandas as pd
import numpy as np

from monet import util
from ..ontology import GOTerm, GeneOntology, GOAnnotation
from . import GeneSet, GeneSetCollection

_LOGGER = logging.getLogger(__name__)


def get_gene_sets(
        go_annotations: Iterable[GOAnnotation],
        min_genes: int = None, max_genes: int = None) -> GeneSetCollection:
    """Generate a list of gene sets from a collection of GO annotations.

    Each gene set corresponds to all genes annotated with a certain GO term.
    """
    go_term_genes = OrderedDict()
    term_ids = {}
    for ann in go_annotations:
        term_ids[ann.go_term.id] = ann.go_term
        try:
            go_term_genes[ann.go_term.id].append(ann.db_symbol)
        except KeyError:
            go_term_genes[ann.go_term.id] = [ann.db_symbol]
    
    go_term_genes = OrderedDict(sorted(go_term_genes.items()))
    gene_sets = []
    for tid, genes in go_term_genes.items():
        genes = sorted(set(genes))
        if (min_genes is None or len(genes) >= min_genes) and \
                (max_genes is None or len(genes) <= max_genes):                
            go_term = term_ids[tid]
            gs = GeneSet(id=tid, name=go_term.name, genes=genes,
                        source='GO',
                        collection=go_term.domain_short,
                        description=go_term.definition)
            gene_sets.append(gs)
    _LOGGER.info('Extracted %d gene sets.', len(gene_sets))
    gene_sets = GeneSetCollection(gene_sets)
    return gene_sets
