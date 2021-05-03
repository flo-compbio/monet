from typing import List
from pkg_resources import resource_filename


def get_mitochondrial_genes(species: str = 'human') -> List[str]:
    path = resource_filename(
        'monet', 'data/gene_lists/mitochondrial_%s.txt' % species)
    with open(path) as fh:
        return fh.read().split('\n')

def get_ribosomal_genes(species: str = 'human') -> List[str]:
    path = resource_filename(
        'monet', 'data/gene_lists/ribosomal_%s.txt' % species)
    with open(path) as fh:
        return fh.read().split('\n')
