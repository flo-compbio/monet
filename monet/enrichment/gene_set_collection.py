# Copyright (c) 2015, 2016, 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `GeneSetCollection` class.

Class supports unicode using UTF-8.

"""

import os
import io
import logging
import hashlib
import csv
from collections import OrderedDict
from typing import List, Iterable, Dict

import numpy as np

# import unicodecsv as csv

from . import GeneSet

logger = logging.getLogger(__name__)


class GeneSetCollection:
    """A collection of gene sets.

    This is a class that basically just contains a list of gene sets, and
    supports different ways of accessing individual gene sets. The gene sets
    are ordered, so each gene set has a unique position (index) in the
    database.

    Parameters
    ----------
    gene_sets: list or tuple of `GeneSet`
        See :attr:`gene_sets` attribute.

    Attributes
    ----------
    gene_sets: tuple of `GeneSet`
        The list of gene sets in the database. Note that this is a read-only
        property.
    """
    def __init__(self, gene_sets: Iterable[GeneSet]):
        
        gene_sets = list(gene_sets)

        # make sure all IDs are unique
        all_ids = [gs.id for gs in gene_sets]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError('Cannot create GeneSetCollection:'
                             'gene set IDs are not unique!')

        self._gene_sets = OrderedDict([gs.id, gs] for gs in gene_sets)
        self._gene_set_ids = list(self._gene_sets.keys())
        self._gene_set_indices = OrderedDict(
            [gs.id, i] for i, gs in enumerate(self._gene_sets.values())
        )

    def __repr__(self):
        return '<%s object (n=%d; hash=%s)>' \
                % (self.__class__.__name__, self.n, self.hash)

    def __str__(self):
        return '<%s object (n=%d)>' % (self.__class__.__name__, self.n)

    def __len__(self):
        return len(self._gene_sets)

    def __iter__(self):
        return iter(self._gene_sets.values())

    def __getitem__(self, key):
        """Simple interface for querying the database.

        Depending on whether key is an integer or not, look up a gene set
        either by index, or by ID.
        """
        if isinstance(key, (int, np.integer)):
            return self.get_by_index(key)
        else:
            return self.get_by_id(key)

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return self.__dict__ == other.__dict__
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def hash(self):
        data = ';'.join(repr(gs) for gs in self.gene_sets)
        return str(hashlib.md5(data.encode('ascii')).hexdigest())

    @property
    def gene_sets(self):
        return list(self._gene_sets.values())

    @property
    def n(self):
        """The number of gene sets in the database."""
        return len(self)

    def add_gene_set(self, gs: GeneSet, overwrite: bool = False):
        if gs.id in self._gene_sets and not overwrite:
            raise ValueError('Gene set with this ID already exists, and '
                             '`overwrite` was not set to ``True``.')
        self._gene_sets[gs.id] = gs
        self._gene_set_ids.append(gs)
        self._gene_set_indices[gs.id] = len(self)-1

    def get_by_id(self, id_: str) -> GeneSet:
        """Look up a gene set by its ID.

        Parameters
        ----------
        id_: str
            The ID of the gene set.

        Returns
        -------
        GeneSet
            The gene set.

        Raises
        ------
        ValueError
            If the given ID is not in the database.
        """
        try:
            return self._gene_sets[id_]
        except KeyError:
            raise ValueError('No gene set with ID "%s"!' % id_)

    def get_by_index(self, i: int) -> GeneSet:
        """Look up a gene set by its index.

        Parameters
        ----------
        i: int
            The index of the gene set.

        Returns
        -------
        GeneSet
            The gene set.

        Raises
        ------
        ValueError
            If the given index is out of bounds.
        """
        if i >= self.n:
            raise ValueError('Index %d out of bounds ' % i +
                             'for database with %d gene sets.' % self.n)
        return self._gene_sets[self._gene_set_ids[i]]

    def index(self, id_: str):
        """Get the index corresponding to a gene set, identified by its ID.

        Parameters
        ----------
        id_: str
            The ID of the gene set.

        Returns
        -------
        int
            The index of the gene set.

        Raises
        ------
        ValueError
            If the given ID is not in the database.
        """
        try:
            return self._gene_set_indices[id_]
        except KeyError:
            raise ValueError('No gene set with ID "%s"!' % id_)

    @classmethod
    def load_tsv(cls, fpath: str, encoding: str = 'utf-8'):
        """Read a gene set database from a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path name of the the file.
        encoding: str
            The encoding of the text file.

        Returns
        -------
        None
        """
        gene_sets = []
        n = 0
        with open(fpath, 'r', encoding=encoding) as fh:
            reader = csv.reader(fh, dialect='excel-tab')
            for l in reader:
                n += 1
                gs = GeneSet.from_list(l)
                gene_sets.append(gs)
        logger.debug('Read %d gene sets.', n)
        logger.debug('Size of gene set list: %d', len(gene_sets))
        return cls(gene_sets)


    def save_tsv(self, fpath) -> None:
        """Save the database to a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path name of the file.

        Returns
        -------
        None
        """
        with open(fpath, 'w') as ofh:
            writer = csv.writer(
                ofh, dialect='excel-tab',
                quoting=csv.QUOTE_NONE, lineterminator=os.linesep
            )
            for gs in self._gene_sets.values():
                writer.writerow(gs.to_list())

    @classmethod
    def load_msigdb_xml(cls, fpath: str, entrez2gene: Dict[str, str],
                        species: str = None):  # pragma: no cover
        """Load the complete MSigDB database from an XML file.

        The XML file can be downloaded from here:
        http://software.broadinstitute.org/gsea/msigdb/download_file.jsp?filePath=/resources/msigdb/5.0/msigdb_v5.0.xml

        Parameters
        ----------
        path: str
            The path name of the XML file.
        entrez2gene: dict or OrderedDict (str: str)
            A dictionary mapping Entrez Gene IDs to gene symbols (names).
        species: str, optional
            A species name (e.g., "Homo_sapiens"). Only gene sets for that
            species will be retained. (None)

        Returns
        -------
        GeneSetCollection
            The gene set database containing the MSigDB gene sets.
        """

        # note: is XML file really encoded in UTF-8?
        import xmltodict

        assert species is None or isinstance(species, str)

        logger.debug('Path: %s', fpath)
        logger.debug('entrez2gene type: %s', str(type(entrez2gene)))

        i = [0]
        gene_sets = []

        total_gs = [0]
        total_genes = [0]

        species_excl = [0]
        unknown_entrezid = [0]

        src = 'MSigDB'

        def handle_item(pth, item):
            # callback function for xmltodict.parse()

            total_gs[0] += 1
            data = pth[1][1]

            spec = data['ORGANISM']
            # filter by species
            if species is not None and spec != species:
                species_excl[0] += 1
                return True

            id_ = data['SYSTEMATIC_NAME']
            name = data['STANDARD_NAME']
            coll = data['CATEGORY_CODE']
            desc = data['DESCRIPTION_BRIEF']
            entrez = data['MEMBERS_EZID'].split(',')

            genes = []
            for e in entrez:
                total_genes[0] += 1
                try:
                    genes.append(entrez2gene[e])
                except KeyError:
                    unknown_entrezid[0] += 1

            if not genes:
                logger.warning('Gene set "%s" (%s) has no known genes!',
                               name, id_)
                return True

            gs = GeneSet(id_, name, genes, source=src,
                         collection=coll, description=desc)
            gene_sets.append(gs)
            i[0] += 1
            return True

        # parse the XML file using the xmltodict package
        with io.open(fpath, 'rb') as fh:
            xmltodict.parse(fh.read(), encoding='UTF-8', item_depth=2,
                            item_callback=handle_item)

        # report some statistics
        if species_excl[0] > 0:
            kept = total_gs[0] - species_excl[0]
            perc = 100 * (kept / float(total_gs[0]))
            logger.info('%d of all %d gene sets (%.1f %%) belonged to the '
                        'specified species.', kept, total_gs[0], perc)

        if unknown_entrezid[0] > 0:
            unkn = unknown_entrezid[0]
            # known = total_genes[0] - unknown_entrezid[0]
            perc = 100 * (unkn / float(total_genes[0]))
            logger.warning('%d of a total of %d genes (%.1f %%) had an ' +
                           'unknown Entrez ID.', unkn, total_genes[0], perc)

        logger.info('Parsed %d entries, resulting in %d gene sets.',
                    total_gs[0], len(gene_sets))

        return cls(gene_sets)
