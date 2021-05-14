# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2015, 2016, 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `GeneOntology` class.
"""

from typing import Iterable
import gzip
import hashlib
# import re
# import sys
import logging
import os
# import bisect

from collections import OrderedDict
import csv

from .. import util
from . import GOTerm

import pickle

_LOGGER = logging.getLogger(__name__)


class GeneOntology:
    """A Gene Ontology.

    This class provides functions for parsing text files describing the Gene
    Ontology, and for accessing information about specific GO terms.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    get_term_by_id(id_)
        Return the term with the given term ID as a `GOTerm` object.
    get_term_by_name(name)
        Return the term with the given name as a `GOTerm` object.
    save(ofn, compress=False)
        Stores the GOParser object as a `pickle` file. If ``compress`` is set
        to True, the object is stored as a gzip'ed pickle file.
    load(fn)
        Loads the `GOParser` object from a `pickle` file. Gzip compression is
        detected automatically.

    Examples
    --------
    The following example assumes that the Gene Ontology OBO file has been
    downloaded.
    >>> from monet.ontology import GeneOntology
    >>> ontology = GeneOntology.load_obo('go-basic.obo')
    """
    def __init__(
            self,
            terms: Iterable[GOTerm] = None,
            syn2id: dict = None,
            alt_id: dict = None,
            name2id: dict = None):

        if terms is None:
            terms = []

        if syn2id is None:
            syn2id = {}

        if alt_id is None:
            alt_id = {}

        if name2id is None:
            name2id = {}

        term_dict = {}
        for t in terms:
            term_dict[t.id] = t
        self._term_dict = term_dict

        self.syn2id = syn2id
        self.alt_id = alt_id
        self.name2id = name2id
        self._flattened = False

    def __repr__(self):
        return '<%s instance (%d GO terms, hash="%s")>' \
               % (self.__class__.__name__, len(self), self.hash)

    def __str__(self):
        return '<%s instance with %d GO terms>' \
               % (self.__class__.__name__, len(self))

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return repr(self) == repr(other)
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, key):
        return self._term_dict[key]

    def __setitem__(self, key, value):
        assert isinstance(value, GOTerm)
        self._term_dict[key] = value

    def __delitem__(self, key):
        del self._term_dict[key]

    def __len__(self):
        return len(self._term_dict)

    def __contains__(self, key):
        return key in self._term_dict

    def __iter__(self):
        return iter(self._term_dict.values())

    @property
    def hash(self):
        data_str = ';'.join(
            [repr(self._term_dict[id_]) for id_ in sorted(self._term_dict.keys())] +
            [repr(var) for var in [self.syn2id, self.alt_id, self.name2id]]
        )
        data = data_str.encode('UTF-8')
        return str(hashlib.md5(data).hexdigest())

    @property
    def flattened(self):
        return self._flattened

    def save_pickle(self, fpath: str, compress: bool = False) -> None:
        """Serialize the current `GOParser` object and store it in a pickle file.

        Parameters
        ----------
        path: str
            Path of the output file.
        compress: bool, optional
            Whether to compress the file using gzip.

        Returns
        -------
        None

        Notes
        -----
        Compression with gzip is significantly slower than storing the file
        in uncompressed form.
        """
        _LOGGER.info('Writing pickle to "%s"...', fpath)
        if compress:
            with gzip.open(fpath, 'wb') as ofh:
                pickle.dump(self, ofh, pickle.HIGHEST_PROTOCOL)
        else:
            with open(fpath, 'wb') as ofh:
                pickle.dump(self, ofh, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_pickle(fpath: str):
        """Load a GOParser object from a pickle file.

        The function automatically detects whether the file is compressed
        with gzip.

        Parameters
        ----------
        fpath: str
            Path of the pickle file.

        Returns
        -------
        `GOParser`
            The GOParser object stored in the pickle file.
        """
        if util.is_gzip_file(fpath):
            with gzip.open(fpath, 'rb') as fh:
                parser = pickle.load(fh)
        else:
            with open(fpath, 'rb') as fh:
                parser = pickle.load(fh)
        return parser


    def get_term_by_id(self, id_: str) -> GOTerm:
        """Get the GO term corresponding to the given GO term ID.

        Parameters
        ----------
        id_: str
            A GO term ID.

        Returns
        -------
        `GOTerm`
            The GO term corresponding to the given ID.
        """
        return self[id_]


    def get_term_by_acc(self, acc: int) -> GOTerm:
        """Get the GO term corresponding to the given GO term accession number.

        Parameters
        ----------
        acc: int
            The GO term accession number.

        Returns
        -------
        `GOTerm`
            The GO term corresponding to the given accession number.
        """
        return self[GOTerm.acc2id(acc)]

    def get_term_by_name(self, name: str) -> GOTerm:
        """Get the GO term with the given GO term name.

        If the given name is not associated with any GO term, the function will
        search for it among synonyms.

        Parameters
        ----------
        name: str
            The name of the GO term.

        Returns
        -------
        `GOTerm`
            The GO term with the given name.

        Raises
        ------
        ValueError
            If the given name is found neither among the GO term names, nor
            among synonyms.
        """
        term = None
        try:
            term = self._term_dict[self.name2id[name]]
        except KeyError:
            try:
                term = self._term_dict[self.syn2id[name]]
            except KeyError:
                pass
            else:
                _LOGGER.info('GO term name "%s" is a synonym for "%s".',
                            name, term.name)

        if term is None:
            raise ValueError('GO term name "%s" not found!' % name)

        return term


    @classmethod
    def load_obo(
            cls,
            fpath: str,
            flatten: bool = True,
            part_of_cc_only: bool = False):
        """ Parse an OBO file and store GO term information.

        Parameters
        ----------
        fpath: str
            Path of the OBO file.
        flatten: bool, optional
            If set to False, do not generate a list of all ancestors and
            descendants for each GO term.
        part_of_cc_only: bool, optional
            Legacy parameter for backwards compatibility. If set to True,
            ignore ``part_of`` relations outside the ``cellular_component``
            domain.

        Notes
        -----
        The OBO file must end with a line break.
        """

        name2id = {}
        alt_id = {}
        syn2id = {}
        terms = []

        fpath_expanded = os.path.expanduser(fpath)

        with open(fpath_expanded) as fh:
            n = 0
            while True:
                try:
                    nextline = next(fh)
                except StopIteration:
                    break
                if nextline == '[Term]\n':
                    n += 1
                    id_ = next(fh)[4:-1]
                    # acc = get_acc(id_)
                    name = next(fh)[6:-1]
                    name2id[name] = id_
                    domain = next(fh)[11:-1]
                    def_ = None
                    is_a = set()
                    part_of = set()
                    l = next(fh)
                    while l != '\n':
                        if l.startswith('alt_id:'):
                            alt_id[l[8:-1]] = id_
                        elif l.startswith('def: '):
                            idx = l[6:].index('"')
                            def_ = l[6:(idx+6)]
                        elif l.startswith('is_a:'):
                            is_a.add(l[6:16])
                        elif l.startswith('synonym:'):
                            idx = l[10:].index('"')
                            if l[(10+idx+2):].startswith("EXACT"):
                                s = l[10:(10+idx)]
                                syn2id[s] = id_
                        elif l.startswith('relationship: part_of'):
                            if part_of_cc_only:
                                if domain == 'cellular_component':
                                    part_of.add(l[22:32])
                            else:
                                part_of.add(l[22:32])
                        l = next(fh)
                    assert def_ is not None
                    terms.append(
                        GOTerm(id_, name, domain, def_, is_a, part_of))

        _LOGGER.info('Parsed %d GO term definitions.', n)

        ontology = cls(terms, syn2id, alt_id, name2id)

        # store children and parts
        _LOGGER.info('Adding child and part relationships...')
        for term in ontology:
            for parent in term.is_a:
                ontology[parent].children.add(term.id)
            for whole in term.part_of:
                ontology[whole].parts.add(term.id)

        if flatten:
            _LOGGER.info('Flattening ancestors...')
            ontology._flatten_ancestors()
            _LOGGER.info('Flattening descendants...')
            ontology._flatten_descendants()
            ontology._flattened = True

        return ontology


    def _flatten_ancestors(self, include_part_of: bool = True):
        """Determines and stores all ancestors of each GO term.

        Parameters
        ----------
        include_part_of: bool, optional
            Whether to include ``part_of`` relations in determining
            ancestors.

        Returns
        -------
        None
        """
        def get_all_ancestors(term):
            ancestors = set()
            for id_ in term.is_a:
                ancestors.add(id_)
                ancestors.update(get_all_ancestors(self[id_]))
            if include_part_of:
                for id_ in term.part_of:
                    ancestors.add(id_)
                    ancestors.update(get_all_ancestors(self[id_]))
            return ancestors

        for term in self:
            term.ancestors = get_all_ancestors(term)


    def _flatten_descendants(self, include_parts: bool = True):
        """Determines and stores all descendants of each GO term.

        Parameters
        ----------
        include_parts: bool, optional
            Whether to include ``part_of`` relations in determining
            descendants.

        Returns
        -------
        None
        """
        def get_all_descendants(term):
            descendants = set()
            for id_ in term.children:
                descendants.add(id_)
                descendants.update(get_all_descendants(self[id_]))
            if include_parts:
                for id_ in term.parts:
                    descendants.add(id_)
                    descendants.update(get_all_descendants(self[id_]))
            return descendants

        for term in self:
            term.descendants = get_all_descendants(term)
