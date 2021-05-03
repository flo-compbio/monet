# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2015, 2016, 2020 Florian Wagner
#
# This file is part of Monet.

import logging
from typing import Iterable, List

import pandas as pd

from . import GeneOntology, GOAnnotation

_LOGGER = logging.getLogger(__name__)


def get_go_annotations(
        fpath: str,
        ontology: GeneOntology,
        valid_genes: Iterable[str] = None,
        db: str = None,
        ev_codes: Iterable[str] = None) -> List[GOAnnotation]:
    """Parse a GAF 2.1 file containing GO annotations.
    
    Parameters
    ----------
    fpath : str or buffer
        The GAF file.
    ontology : `GeneOntology`
        The Gene Ontology.
    valid_genes : Iterable of str, optional
        A list of valid gene names. [None]
    db : str, optional
        Select only annotations with this "DB"" value. [None]
    ev_codes : str or set of str, optional
        Select only annotations with this/these evidence codes. [None]
    
    Returns
    -------
    list of `GOAnnotation`
        The list of GO annotations.
    """

    # use pandas to parse the file quickly
    df = pd.read_csv(fpath, sep='\t', comment='!', header=None, dtype=str)

    # replace pandas' NaNs with empty strings
    df.fillna('', inplace=True)

    # exclude annotations with unknown Gene Ontology terms
    all_go_term_ids = set(ontology._term_dict.keys())
    sel = df.iloc[:, 4].isin(all_go_term_ids)
    _LOGGER.info(
        'Ignoring %d / %d annotations (%.1f %%) with unknown GO terms.',
        (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
    df = df.loc[sel]

    # filter rows for valid genes
    if valid_genes is not None:
        sel = df.iloc[:, 2].isin(valid_genes)
        _LOGGER.info(
            'Ignoring %d / %d annotations (%.1f %%) with unknown genes.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # filter rows for DB value
    if db is not None:
        sel = (df.iloc[:, 0] == db)
        _LOGGER.info(
            'Excluding %d / %d annotations (%.1f %%) with wrong DB values.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # filter rows for evidence value
    if ev_codes is not None:
        sel = (df.iloc[:, 6].isin(ev_codes))
        _LOGGER.info(
            'Excluding %d / %d annotations (%.1f %%) based on evidence code.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # convert each row into a GOAnnotation object
    go_annotations = []
    for i, l in df.iterrows():
        ann = GOAnnotation.from_list(ontology, l.tolist())
        go_annotations.append(ann)
    _LOGGER.info('Read %d GO annotations.', len(go_annotations))

    return go_annotations
