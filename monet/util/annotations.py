# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Utility functions for working with gene annotations."""

import re
from typing import Dict, Iterable
import logging

import pandas as pd

_LOGGER = logging.getLogger(__name__)

# don't split on escaped semicolons ("\;")
# (= negative look-behind)
_ATT_PATTERN = re.compile(r'(?<!\\)\s*;\s*')


def _parse_attributes(att_string: str) -> Dict[str, str]:
    attributes = {}
    kv_pairs = _ATT_PATTERN.split(att_string)
    for keyval_str in kv_pairs:
        keyval = keyval_str.split(' ', maxsplit=1)
        if len(keyval) == 2:
            k, v = keyval
            attributes[k] = v.strip('""')
    return attributes


def get_ensembl_genes(
        annotation_file: str,
        chunksize: int = 100000,
        gene_types: Iterable[str] = None) -> pd.DataFrame:

    if gene_types is None:
        _LOGGER.info('Extract all genes from GTF file...')
    else:
        _LOGGER.info('Extracting selected genes from GTF file...')
        _LOGGER.info('Selected gene types: %s',
                     ', '.join(sorted(gene_types)))

    if gene_types is not None:
        gene_types = set(gene_types)

    parser = pd.read_csv(
        annotation_file,
        chunksize=chunksize,
        encoding='ascii',
        sep='\t',
        header=None,
        comment='#',
        dtype={0: str})

    num_chunks = 0
    num_lines = 0
    num_gene_lines = 0

    data = []

    for j, df in enumerate(parser):

        num_chunks += 1
        num_lines += (df.shape[0])

        # select "gene" rows
        df = df.loc[df.iloc[:, 2] == 'gene']
        num_gene_lines += (df.shape[0])

        for i, row in df.iterrows():

            # parse attributes
            attributes = _parse_attributes(row.iloc[8])

            if gene_types is None or \
                    attributes['gene_biotype'] in gene_types:

                chromosome = row.iloc[0]
                type_ = attributes['gene_biotype']
                ensembl_id = attributes['gene_id']
                name = attributes['gene_name']

                d = [ensembl_id, name, chromosome, type_]
                data.append(d)

    columns = ['Ensembl ID', 'Name', 'Chromosome', 'Type']
    genes = pd.DataFrame(data=data, columns=columns)
    genes.sort_values('Name', inplace=True)
    genes.set_index('Ensembl ID', inplace=True)
    num_valid_genes = genes.shape[0]

    _LOGGER.info('Parsed GTF file with %d lines (%d chunks).',
                 num_lines, num_chunks)
    _LOGGER.info('Found %d lines with gene annotations and %d valid genes.',
                 num_gene_lines, num_valid_genes)

    return genes


def get_ensembl_protein_coding_genes(annotation_file: str, **kwargs) \
        -> pd.DataFrame:
    
    _LOGGER.info('Extracting all protein-coding genes...')
    
    additional_gene_types = set(kwargs.pop('gene_types', set()))
    if additional_gene_types:
        _LOGGER.info('Additional gene types: %s',
                     ', '.join(sorted(additional_gene_types)))    
    
    gene_types = {'protein_coding'} | \
        {'IG_V_gene', 'IG_D_gene', 'IG_J_gene', 'IG_C_gene'} | \
        {'TR_V_gene', 'TR_D_gene', 'TR_J_gene', 'TR_C_gene'}
    
    kwargs['gene_types'] = gene_types | additional_gene_types
    
    genes = get_ensembl_genes(annotation_file, **kwargs)

    return genes


def get_ensembl_immune_genes(annotation_file: str, **kwargs) -> pd.DataFrame:

    _LOGGER.info('Extracting all protein-coding genes, '
                 'plus immune-specific genes...')

    additional_gene_types = set(kwargs.pop('gene_types', set()))
    if additional_gene_types:
        _LOGGER.info('Additional gene types: %s',
                     ', '.join(sorted(additional_gene_types)))
    
    gene_types = \
        {'protein_coding'} | \
        {'IG_V_gene', 'IG_D_gene', 'IG_J_gene', 'IG_C_gene'} | \
        {'TR_V_gene', 'TR_D_gene', 'TR_J_gene', 'TR_C_gene'}

    kwargs['gene_types'] = gene_types | additional_gene_types    
    
    genes = get_ensembl_genes(annotation_file, **kwargs)

    return genes
