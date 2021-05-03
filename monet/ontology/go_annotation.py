# Copyright (c) 2015, 2016, 2020 Florian Wagner
#
# This file is part of sMonet.

"""Module containing the `GOAnnotation` class."""

import re
import hashlib
from collections import OrderedDict
import logging
import tempfile

from . import GOTerm, GeneOntology

logger = logging.getLogger(__name__)


class GOAnnotation:
    """Class representing an annotation of a gene with a GO term.

    For a list of annotation properties, see the
    `GAF 2.1 file format specification`__. 
    
    __ gafformat_

    Parameters
    ----------
    gene : str
        See :attr:`gene` attribute.
    term: `GOTerm` object
        See :attr:`term` attribute.
    evidence: str
        See :attr:`evidence` attribute.
    db_id: str, optional
        See :attr:`db_id` attribute.
    db_ref: list of str, optional
        See :attr:`db_ref` attribute.
    with_: list of str, optional
        See :attr:`with_` attribute.

    Attributes
    ----------
    gene: str
        The gene that is annotated (e.g., "MYOD1").
    term: `GOTerm` object
        The GO term that the gene is annotated with.
    evidence: str
        The three-letter evidence code of the annotation (e.g., "IDA").
    db_id: str, optional
        Database Object ID of the annotation.
    db_ref: list of str, optional
        DB:Reference of the annotation.
    with_: list of str, optional
        "With" information of the annotation.

    Methods
    -------
    get_gaf_format()
        Return the annotation as a tab-delimited string acccording to the
        `GAF 2.1 file format`__.
    get_pretty_format()
        Return a nicely formatted string representation of the GO annotation.
    __ gafformat_
    .. _gafformat: http://geneontology.org/page/go-annotation-file-gaf-format-21
    """

    _evidence_name = {
        'EXP': 'experiment',
        'IDA': 'direct assay',
        'IPI': 'physical interaction',
        'IMP': 'mutant phenotype',
        'IGI': 'genetic interaction',
        'IEP': 'expression pattern',
        'ISS': 'sequence or structural similarity',
        'ISO': 'sequence orthology',
        'ISA': 'sequence alignment',
        'ISM': 'sequence model',
        'IGC': 'genomic context',
        'IBA': 'biological aspect of ancestor',
        'IBD': 'biological aspect of descendant',
        'IKR': 'key residues',
        'IRD': 'rapid divergence',
        'RCA': 'reviewed computational analysis',
        'TAS': 'traceable author statement',
        'NAS': 'non-traceable author statement',
        'IC' : 'inferred by curator',
        'ND' : 'no biological data available',
        'IEA': 'inferred from electronic annotation'
    }
    """Mapping of the three-letter evidence codes to their full names.
    """

    _evidence_type = {
        'EXP': 'experimental',
        'IDA': 'experimental',
        'IPI': 'experimental',
        'IMP': 'experimental',
        'IGI': 'experimental',
        'IEP': 'experimental',
        'ISS': 'computational',
        'ISO': 'computational',
        'ISA': 'computational',
        'ISM': 'computational',
        'IGC': 'computational',
        'IBA': 'computational',
        'IBD': 'computational',
        'IKR': 'computational',
        'IRD': 'computational',
        'RCA': 'computational',
        'TAS': 'literature',
        'NAS': 'literature',
        'IC' : 'curator',
        'ND' : 'no_data',
        'IEA': 'automatic'
    }
    """Mapping of the three-letter evidence codes to their evidence types.
    """

    _evidence_type_short = {
        'experimental': 'exp.',
        'computational': 'comp.',
        'literature': 'lit.',
        'curator': 'cur.',
        'no_data': 'n.d.',
        'automatic': 'autom.'
    }
    """Mapping of the evidence types to abbreviated forms.
    """

    # uniprot_pattern = re.compile("([A-Z][A-Z0-9]{5})(?:-(\d+))?")

    def __init__(self, db: str, db_id: str, db_symbol: str,
                 go_term: GOTerm, db_ref, ev_code,
                 db_type: str, taxon, date, assigned_by, **kwargs):

        qualifier = kwargs.pop('qualifier', None)
        with_from = kwargs.pop('with_from', None)
        db_name = kwargs.pop('db_name', None)
        db_syn = kwargs.pop('db_syn', None)
        ext = kwargs.pop('ext', None)
        product_id = kwargs.pop('product_id', None)

        ### type checks
        # assert isinstance(db, (str, _oldstr))
        # assert isinstance(db_id, (str, _oldstr))
        # assert isinstance(db_symbol, (str, _oldstr))
        # assert isinstance(go_term, GOTerm)
        # assert isinstance(db_ref, (str, _oldstr)) or \
        #         isinstance(db_ref, Iterable)
        # assert isinstance(ev_code, (str, _oldstr))
        # assert isinstance(db_type, (str, _oldstr))
        # assert isinstance(taxon, (str, _oldstr)) or \
        #         isinstance(taxon, Iterable)
        # assert isinstance(ev_code, (str, _oldstr))
        # assert isinstance(date, (str, _oldstr))
        # assert isinstance(assigned_by, (str, _oldstr))

        # if qualifier is not None:
        #     assert isinstance(qualifier, (str, _oldstr)) or \
        #             isinstance(qualifier, Iterable)
        # if with_from is not None:
        #     assert isinstance(with_from, (str, _oldstr))
        # if db_name is not None:
        #     assert isinstance(db_name, (str, _oldstr))
        # if db_syn is not None:
        #     assert isinstance(db_syn, (str, _oldstr)) or \
        #             isinstance(db_syn, Iterable)
        # if ext is not None:
        #     assert isinstance(ext, (str, _oldstr)) or \
        #             isinstance(ext, Iterable)
        # if product_id is not None:
        #     assert isinstance(db_name, (str, _oldstr))

        ### convert all `None` arguments for attributes with
        #   max(cardinaility) > 0 to empty lists
        if qualifier is None:
            qualifier = []
        # with/from is currently left as string
        # if with_from is None:
        #    with_from = []
        if db_syn is None:
            db_syn = []
        if ext is None:
            ext = [] 

        ### enable flexibility in terms of passing strings or lists
        if isinstance(db_ref, str):
            db_ref = db_ref.split('|')
        if isinstance(taxon, str):
            taxon = taxon.split('|')

        if isinstance(qualifier, str):
            qualifier = qualifier.split('|')
        # with/from is currently left as string
        # if isinstance(with_from, (str, _oldstr)): 
        #     with_from = re.split(',|\|', with_from)
        if isinstance(db_syn, str):
            db_syn = db_syn.split('|')
        if isinstance(ext, str):
            ext = ext.split('|')

        self.db = db
        self.db_id = db_id
        self.db_symbol = db_symbol
        self.go_term = go_term
        self.db_ref = db_ref
        self.ev_code = ev_code
        self.db_type = db_type
        self.taxon = taxon
        self.date = date
        self.assigned_by = assigned_by

        self.qualifier = qualifier
        self.with_from = with_from
        self.db_name = db_name
        self.db_syn = db_syn
        self.ext = ext
        self.product_id = product_id


    def __repr__(self):
        return '<GOAnnotation object (hash=%s)>' % self.hash


    def __str__(self):
        return '<GOAnnotation of "%s" with GO term "%s" (%s)>' \
                % (self.db_symbol, self.go_term.name, self.go_term.id)


    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return repr(self) == repr(other)
        else:
            return NotImplemented


    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        return hash(repr(self))

    @property
    def hash(self):
        data_str = ';'.join([
            str(repr(var)) for var in [
                self.db,
                self.db_id,
                self.db_symbol,
                self.go_term,
                self.db_ref,
                self.ev_code,
                self.db_type,
                self.taxon,
                self.date,
                self.assigned_by,

                self.qualifier,
                self.with_from,
                self.db_name,
                self.db_syn,
                self.ext,
                self.product_id,
            ]
        ])
        #data_str += ';'
        #data = data_str.encode('UTF-8') + \
        #    b';'.join([a.tobytes() for a in [
        #        self.Y, self.W
        #    ]])
        data = data_str.encode('UTF-8')
        return str(hashlib.md5(data).hexdigest())


    @property
    def aspect(self):
        if self.go_term.domain == 'biological_process':
            return 'P'
        elif self.go_term.domain == 'molecular_function':
            return 'F'
        elif self.go_term.domain == 'cellular_component':
            return 'C'
        else:
            return None


    @property
    def with_from_list(self):
        """Converts the with_from string to a list."""
        return re.split(',|\|', self.with_from)


    @classmethod
    def from_list(cls, gene_ontology, l):
        """Initialize a `GOAnnotation` object from a list (in GAF2.1 order).
        
        TODO: docstring
        """ 
        assert isinstance(gene_ontology, GeneOntology)
        assert isinstance(l, list)

        assert len(l) == 17

        go_term = gene_ontology[l[4]]

        qualifier = l[3] or []
        with_from = l[7] or None
        db_name = l[9] or None
        db_syn = l[10] or []
        ext = l[15] or []
        product_id = l[16] or None

        annotation = cls(
            db=l[0],
            db_id=l[1],
            db_symbol=l[2],
            go_term=go_term,
            db_ref=l[5],
            ev_code=l[6],
            db_type=l[11],
            taxon=l[12],
            date=l[13],
            assigned_by=l[14],

            qualifier=qualifier,
            with_from=with_from,
            db_name=db_name,
            db_syn=db_syn,
            ext=ext,
            product_id=product_id,
        )
        return annotation


    @property
    def as_list(self):
        """Returns GO annotation as a flat list (in GAF 2.1 format order)."""
        go_id = self.go_term.id
        
        qual_str = '|'.join(self.qualifier)
        db_ref_str = '|'.join(self.db_ref)
        taxon_str = '|'.join(self.taxon)

        # with_from is currently left as a string
        # with_from = '|'.join()
        with_from_str = self.with_from or ''
        db_name_str = self.db_name or ''
        db_syn_str = '|'.join(self.db_syn)
        ext_str = '|'.join(self.ext)
        product_id_str = self.product_id or ''

        l = [
            self.db,
            self.db_id,
            self.db_symbol,
            qual_str,
            go_id,
            db_ref_str,
            self.ev_code,
            with_from_str,
            self.aspect,
            db_name_str,
            db_syn_str,
            self.db_type,
            taxon_str,
            self.date,
            self.assigned_by,
            ext_str,
            product_id_str,
        ]
        return l


    def to_list(self):
        ### TODO: docstring
        return self.as_list

    def get_pretty_format(self):
        """Returns a nicely formatted string with the annotation information.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        str
            The formatted string.
        """
        pretty = ('Annotation of gene "%s" with GO term "%s"'
                  '(%s, reference: %s)'
                  % (self.db_symbol,self.go_term.get_pretty_format(),
                     self.ev_code,
                    '|'.join(self.db_ref)))
        return pretty


def get_latest_goa_release(species):
    """Query the UniProt-GOA FTP server to determine the latest release.

    Parameters
    ----------
    species : str
        The name of the species.

    Returns
    -------
    (int, str)
        The version number and date of the latest release.
    """
    assert isinstance(species, str)

    return None

    #with tempfile.NamedTemporaryFile as tf:
    #    misc.ftp_download('ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/'
    #                      'current_release_numbers.txt',
    #                      tf.name)
    #    df = pd.read_csv(tf.name, sep='\t', comment='!', header=None)

    #species_rel = {}
    #for i, row in df.iterrows():
    #    species_rel[row[0]] = (int(row[1]), str(row[2]))
    #    logger.debug(str(species_rel[row[0]]))

    #return species_rel[species]

