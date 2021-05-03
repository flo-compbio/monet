# Copyright (c) 2015, 2016, 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `GeneSet` class."""

import hashlib
from typing import List, Iterable


class GeneSet:
    """A gene set.

    A gene set is just what the name implies: A set of genes. Usually, gene
    sets are used to group genes that share a certain property (e.g., genes
    that perform related functions, or genes that are frequently co-expressed).
    The genes in the gene set are not ordered.

    GeneSet instances are hashable and should therefore be considered to be
    immutable.
    
    Parameters
    ----------
    id: str
        See :attr:`id` attribute.
    name: str
        See :attr:`name` attribute.
    genes: set, list or tuple of str
        See :attr:`genes` attribute.
    source: str, optional
        See :attr:`source` attribute. (None)
    collection: str, optional
        See :attr:`collection` attribute. (None)
    description: str, optional
        See :attr:`description` attribute. (None)

    Attributes
    ----------
    id_: str
        The (unique) ID of the gene set.
    name: str
        The name of the gene set.
    genes: set of str
        The list of genes in the gene set.
    source: None or str
        The source / origin of the gene set (e.g., "MSigDB")
    collection: None or str
        The collection that the gene set belongs to (e.g., "c4" for gene sets
        from MSigDB).
    description: None or str
        The description of the gene set.
    """
    def __init__(self, id: str, name: str, genes: Iterable[str],
                 source: str = None, collection: str = None,
                 description: str = None):

        self._id = id
        self._name = name
        self._genes = frozenset(genes)
        self._source = source
        self._collection = collection
        self._description = description

    @property
    def _gene_str(self):
        return ', '.join('"%s"' % g for g in sorted(self._genes))

    @property
    def _source_str(self):
        return '"%s"' % self._source \
            if self._source is not None else 'None'

    @property
    def _coll_str(self):
        return '"%s"' % self._collection \
            if self._collection is not None else 'None'

    @property
    def _desc_str(self):
        return '"%s"' % self._description \
            if self._description is not None else 'None'

    def __repr__(self):
        return ('<%s instance (id="%s", name="%s", genes=[%s], source=%s, '
                'collection=%s, description=%s)'
                % (self.__class__.__name__,
                   self._id, self._name, self._gene_str,
                   self._source_str, self._coll_str, self._desc_str))

    def __str__(self):
        return ('<%s "%s" (id=%s, source=%s, collection=%s, size=%d'
                % (self.__class__.__name__, self._name,
                   self._id, self._source_str, self._coll_str, self.size))

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return repr(self) == repr(other)
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def _data(self):
        data_str = ';'.join([
            str(repr(var)) for var in
            [self._id, self._name, self._genes,
             self._source, self._collection, self._description]
        ])
        data = data_str.encode('UTF-8')
        return data

    def __hash__(self):
        return hash(self._data)

    @property
    def hash(self):
        """MD5 hash value for the gene set."""
        return str(hashlib.md5(self._data).hexdigest())

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def genes(self):
        return self._genes

    @property
    def source(self):
        return self._source

    @property
    def collection(self):
        return self._collection

    @property
    def description(self):
        return self._description

    @property
    def size(self):
        """The size of the gene set (i.e., the number of genes in it)."""
        return len(self._genes)

    def to_list(self) -> List[str]:
        """Converts the GeneSet object to a flat list of strings.

        Note: see also :meth:`from_list`.

        Parameters
        ----------

        Returns
        -------
        list of str
            The data from the GeneSet object as a flat list.
        """
        src = self._source or ''
        coll = self._collection or ''
        desc = self._description or ''

        l = [self._id, src, coll, self._name,
             ','.join(sorted(self._genes)), desc]
        return l

    @classmethod
    def from_list(cls, l: Iterable[str]):
        """Generate an GeneSet object from a list of strings.

        Note: See also :meth:`to_list`.

        Parameters
        ----------
        l: list or tuple of str
            A list of strings representing gene set ID, name, genes,
            source, collection, and description. The genes must be
            comma-separated. See also :meth:`to_list`.

        Returns
        -------
        `genometools.basic.GeneSet`
            The gene set.
        """
        id_ = l[0]
        name = l[3]
        genes = l[4].split(',')

        src = l[1] or None
        coll = l[2] or None
        desc = l[5] or None

        return cls(id_, name, genes, src, coll, desc)
