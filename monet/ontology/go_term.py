# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `GOTerm` class."""

import re
from collections import OrderedDict


class GOTerm:
    """A GO term.

    This class is used by the `GeneOntology` class to store all parsed GO term data.

    Parameters
    ----------
    id_: str
        See :attr:`id` attribute.
    name: str
        See :attr:`name` attribute.
    definition: str, optional
        See :attr:`definition` attribute.
    domain: str, optional
        See :attr:`domain` attribute.
    is_a: Iterable of str, optional
        See :attr:`is_a` attribute.
    part_of: Iterable of str, optional.
        See :attr:`part_of` attribute.

    Attributes
    ----------
    id: str
        The ID of the GO term.
    name: str
        The name of the GO term.
    domain: str
        The domain of the GO term (e.g., "biological_process").
    definition: str
        The definition (description) of the GO term.
    is_a: set of str
        Set of GO term IDs that this GO term is a "subtype" of.
    part_of: set of str
        Set of GO term IDs that this GO term is a "part" of.
    ancestors: set of str
        Set of GO term IDs that are "ancestors" of this GO term.
    children: set of str
        Set of GO term IDs that are "children" of this GO term.
    parts: set of str
        Set of GO term IDs that are "parts" of this GO term.
    descendants: set of str
        Set of GO terms IDs that are "descendants" of this GO term.

    Methods
    -------
    get_pretty_format(omit_acc=False, max_name_length=0, abbreviate=True)
        Returns a formatted version of the GO term name and ID.
    """

    _short_domain = {
        'biological_process': 'BP',
        'molecular_function': 'MF',
        'cellular_component': 'CC'
    }
    """Dictionary representing the abbreviations of the Gene Ontology domains.
    """

    _abbrev = [
        ('positive ', 'pos. '),
        ('negative ', 'neg. '),
        ('interferon-', 'IFN-'),
        ('proliferation', 'prolif.'),
        ('signaling', 'signal.')
    ]
    """List of tuples defining abbreviations to use in GO term names.
    """

    def __init__(self, id_, name,
                 domain=None, definition=None,
                 is_a=None, part_of=None):

        if is_a is None:
            is_a = []

        if part_of is None:
            part_of = []

        assert isinstance(id_, str)
        assert isinstance(name, str)

        if domain is not None:
            assert isinstance(domain, str)

        if definition is not None:
            assert isinstance(definition, str)

        #assert isinstance(is_a, Iterable)
        #assert isinstance(part_of, Iterable)

        self.id = id_  # unique identifier
        self.name = name
        self.domain = domain
        self.definition = definition

        # to store immediate parents/wholes
        self.is_a = set(is_a)
        self.part_of = set(part_of)

        # to store immediate children/parts
        self.children = set()
        self.parts = set()

        # to store all descendants/ancestors
        self.descendants = None
        self.ancestors = None

    def __repr__(self):
        # The ID uniquely identifies the term
        return '<GOTerm instance (id=%s)>' % self.id

    def __str__(self):
        return '<GOTerm "%s">' % self.get_pretty_format()

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return repr(self) == repr(other)
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def id2acc(id_):
        """Converts a GO term ID to an accession number.

        Parameters
        ----------
        id_: str
            A GO term ID.

        Returns
        -------
        int
            The accession number corresponding to the GO term ID.

        """
        return int(id_[3:])

    @staticmethod
    def acc2id(acc):
        """Converts a GO term accession number to an ID.

        Parameters
        ----------
        acc: int
            A GO term accession number.

        Returns
        -------
        str
            The ID corresponding to the GO term accession number.
        """
        return 'GO:%07d' % acc

    @property
    def acc(self):
        """Returns the GO term accession number (part of the ID)."""
        return self.id2acc(self.id)

    @property
    def domain_short(self):
        return self._short_domain[self.domain]

    def get_pretty_format(self, include_id=True, max_name_length=0,
                          abbreviate=True):
        """Returns a nicely formatted string with the GO term information.

        Parameters
        ----------
        include_id: bool, optional
            Include the GO term ID.
        max_name_length: int, optional
            Truncate the formatted string so that its total length does not
            exceed this value.
        abbreviate: bool, optional
            Do not use abberviations (see ``_abbrev``) to shorten the GO term
            name.

        Returns
        -------
        str
            The formatted string.
        """
        name = self.name
        if abbreviate:
            for abb in self._abbrev:
                name = re.sub(abb[0], abb[1], name)
        if 3 <= max_name_length < len(name):
            name = name[:(max_name_length-3)] + '...'
        if include_id:
            return "%s: %s (%s)" % (self.domain_short, name, self.id)
        else:
            return "%s: %s" % (self.domain_short, name)
