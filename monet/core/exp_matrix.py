# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

"""Module containing the `ExpMatrix` class."""

import importlib
import logging
import hashlib
import os
import csv
import tarfile
import io
import gzip
from typing import List

import pandas as pd
import numpy as np
import scipy.io
from scipy import sparse

exp_profile = importlib.import_module('.exp_profile', package='monet.core')

_LOGGER = logging.getLogger(__name__)


class ExpMatrix(pd.DataFrame):
    """A gene expression matrix."""

    def __init__(self, *args, genes=None, cells=None, **kwargs):
        
        if genes is not None:
            if 'index' in kwargs or len(args) >= 2:
                raise ValueError(
                    'Providing both `genes` and `index` is redundant!')
            kwargs['index'] = genes

        if cells is not None:
            if 'columns' in kwargs or len(args) >= 3:
                raise ValueError(
                    'Providing both `cells` and `columns` is redundant!')
            kwargs['columns'] = cells

        pd.DataFrame.__init__(self, *args, **kwargs)

        if self.index.name is None:
            self.index.name = 'Genes'

        if self.columns.name is None:
            self.columns.name = 'Cells'


    def __repr__(self):
        return '<%s instance with %d cells and %d genes>' \
               % (self.__class__.__name__, self.num_cells, self.num_genes)

    def __str__(self):
        return super().__repr__()

    @property
    def hash(self) -> str:
        from ..util import calculate_hash
        return calculate_hash(self)

    @property
    def _constructor(self):
        return ExpMatrix
    
    @property
    def _constructor_sliced(self):
        return exp_profile.ExpProfile

    @property
    def num_genes(self):
        """The number of genes."""
        return self.shape[0]

    @property
    def num_cells(self):
        """The number of cells."""
        return self.shape[1]

    @property
    def genes(self):
        """Alias for `DataFrame.index`."""
        return self.index

    @genes.setter
    def genes(self, genes):
        self.index = genes

    @property
    def cells(self):
        """Alias for `DataFrame.columns`."""
        return self.columns

    @cells.setter
    def cells(self, cells):
        self.columns = cells

    @property
    def X(self):
        """The expression data (a cell-by-gene matrix)."""
        return self.values.T
    
    @property
    def median_transcript_count(self):
        """The median transcript count of the cells in the matrix."""
        return float(self.sum(axis=0).median())


    def sort_genes(self, inplace=False):
        """Shortcut for sort_index(axis=0)."""
        return self.sort_index(kind='mergesort', inplace=inplace)


    def sort_cells(self, inplace=False):
        """Shortcut for sort_index(axis=1)."""
        return self.sort_index(axis=1, kind='mergesort', inplace=inplace)


    def scale(self, transcript_count=None, inplace=False):
        """Scale all expression profiles to the same transcript count.

        If `transcript_count` is is not provided, uses the median
        transcript count of all cells in the matrix."""

        num_transcripts = self.sum(axis=0)
        if transcript_count is None:
            transcript_count = float(num_transcripts.median())

        transcript_count = np.array(
            [transcript_count], dtype=self.values.dtype)
        scaled_matrix = (transcript_count / num_transcripts) * self

        if inplace:
            self._update_inplace(scaled_matrix)
            scaled_matrix = self

        return scaled_matrix


    def transform(self, name='freeman-tukey', inplace=False):
        """Apply a transformation to the expression profiles."""

        valid_transforms = ['freeman-tukey', 'anscombe', 'log',
                            'pearson', 'sqrt', 'none', None]

        if name not in valid_transforms:
            valid_transform_str = ', '.join(
                ['"%s"' % trans for trans in valid_transforms[:-1]])
            valid_transform_str = valid_transform_str + \
                ' and "%s".' % valid_transforms[-1]
            raise ValueError(
                '"%s" is not a valid transform name. Choose among %s' \
                % (name, valid_transform_str))

        if name == 'anscombe':
            transformed_matrix = 2 * np.sqrt(self + 3.0/8.0)

        elif name == 'freeman-tukey':
            transformed_matrix = np.sqrt(self) + np.sqrt(self + 1)

        elif name == 'log':
            transformed_matrix = np.log(self + 1)

        elif name == 'pearson':
            transformed_matrix = self.pearson_transform()

        elif name == 'sqrt':
            transformed_matrix = np.sqrt(self)

        elif name == 'none' or name is None:
            transformed_matrix = self

        if inplace:
            self._update_inplace(transformed_matrix)
            transformed_matrix = self

        return transformed_matrix


    def ft_transform(self, inplace: bool = False):
        """Apply the Freeman-Tukey transform to stabilize variance."""
        from ..util import ft_transform
        matrix = ft_transform(self)

        if inplace:
            self._update_inplace(matrix)
            matrix = self

        return matrix


    def pearson_transform(self, inplace: bool = False):
        """Use pearson residuals to stabilize variance."""
        from ..util import pearson_transform

        matrix = pearson_transform(self)

        if inplace:
            self._update_inplace(matrix)
            matrix = self

        return matrix


    def get_variable_genes_fano(self, num_genes: int = 1000,
                           min_exp_thresh: float = 0.0) -> pd.DataFrame:
        from ..util import get_variable_genes_fano

        var_genes, _ = get_variable_genes_fano(
            self, num_genes=num_genes, min_exp_thresh=min_exp_thresh)

        return var_genes


    def get_variable_genes_cv(self, num_genes: int = 1000,
                              min_exp_thresh: float = 0.05) -> pd.DataFrame:
        from ..util import get_variable_genes_cv

        var_genes, _ = get_variable_genes_cv(
            self, num_genes=num_genes, min_exp_thresh=min_exp_thresh)

        return var_genes


    def center_genes(self, inplace: bool = False):
        """Center gene expression values."""
        from ..util import center_genes
        matrix = center_genes(self, inplace=inplace)
        return matrix


    def scale_genes(self, inplace: bool = False):
        """Scale gene expression values so that maximum value is is 1."""
        from ..util import scale_genes
        matrix = scale_genes(self, inplace=inplace)
        return matrix


    def normalize_genes(self, inplace: bool = False):
        """Scale gene expression values to the range [0, 1]."""
        from ..util import normalize_genes
        matrix = normalize_genes(self, inplace=inplace)
        return matrix


    def standardize_genes(self, inplace: bool = False):
        """Convert gene expression values to z-scores."""
        from ..util import standardize_genes
        matrix = standardize_genes(self, inplace=inplace)
        return matrix


    @classmethod
    def from_anndata(cls, adata, dtype=None):
        """Import data from a Scanpy `AnnData` object."""

        genes = adata.var_names.copy()
        genes.name = 'Genes'

        cells = adata.obs_names.copy()
        cells.name = 'Cells'

        data = adata.X.T.copy()

        if dtype is not None:
            # convert data type
            data = data.astype(dtype)

        if sparse.issparse(data):
            # convert from sparse matrix
            data = data.todense()

        matrix = cls(data=data, genes=genes, cells=cells)

        return matrix

    @classmethod
    def load(cls, fpath: str):
        loading_functions = [cls.load_npz, cls.load_anndata]
        matrix = None
        for func in loading_functions:
            try:
                matrix = func(fpath)
            except Exception as e:
                pass
            else:
                break

        if matrix is None:
            raise OSError('Failed to load data!')

        return matrix


    def save(self, fpath: str):
        """Save matrix with automatic file type detection."""

        suffixes = {
            '.npz': self.save_npz,
            '.h5ad': self.save_anndata,
        }

        save_func = None
        for suf, func in suffixes.items():
            if fpath.endswith(suf):
                save_func = func

        if save_func is None:
            _LOGGER.warning(
                'Could not determine file format from file extension. '
                'Will use Monet''s .npz format.')
            save_func = self.save_npz

        save_func(fpath)


    @classmethod
    def load_anndata(cls, fpath: str, dtype=None):
        """Load data from a Scanpy `AnnData` .h5ad file. (requires scanpy!)"""

        try:
            #from anndata import AnnData, read_h5ad
            import anndata as ann
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'You must install the `anndata` package first!') from e

        fpath_expanded = os.path.expanduser(fpath)
        file_size_mb = os.path.getsize(fpath_expanded) / 1e6

        adata = ann.read_h5ad(fpath_expanded)
        matrix = cls.from_anndata(adata, dtype=dtype)

        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            '.h5ad (AnnData) format, %.1f MB (hash: %s).',
            matrix.num_cells, matrix.num_genes, file_size_mb, matrix.hash)

        return matrix


    def to_anndata(self):
        """Export to `AnnData` object. (requires scanpy!)"""

        try:
            from anndata import AnnData
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'You must install the `anndata` package first!') from e

        adata = AnnData(self.T.copy())

        return adata


    def save_anndata(self, fpath: str, compression='gzip', **kwargs) -> None:
        """Save data to a Scanpy `AnnData` .h5ad file. (requires scanpy!)"""

        fpath_expanded = os.path.expanduser(fpath)

        adata = self.to_anndata()
        adata.write_h5ad(fpath_expanded, compression=compression, **kwargs)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Saved expression matrix with %d cells and %d genes -- '
            '.h5ad (AnnData) format, %.1f MB (hash: %s).',
            self.num_cells, self.num_genes, file_size_mb, self.hash)


    @classmethod
    def load_npz(cls, fpath):
        """Load expression matrix from a .npz file."""

        fpath_expanded = os.path.expanduser(fpath)

        data = np.load(fpath_expanded, allow_pickle=True)
        genes = data['genes']
        cells = data['cells']
        data = np.array(data['matrix'].T, order='F', copy=False)
        matrix = cls(data, genes=genes, cells=cells)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            '.npz (Monet) format, %.1f MB (hash: %s).',
            matrix.num_cells, matrix.num_genes, file_size_mb, matrix.hash)

        return matrix


    def save_npz(self, fpath, compressed=True):
        """Save expression matrix as a .npz file."""

        fpath_expanded = os.path.expanduser(fpath)

        data = {}
        data['genes'] = np.array(self.genes.tolist())
        data['cells'] = np.array(self.cells.tolist())
        data['matrix'] = np.array(self.values.T, copy=False)
        if compressed:
            np.savez_compressed(fpath_expanded, **data)
        else:
            np.savez(fpath_expanded, **data)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Saved expression matrix with %d cells and %d genes -- '
            '.npz (Monet) format, %.1f MB (hash: %s).',
            self.num_cells, self.num_genes, file_size_mb, self.hash)


    @classmethod
    def load_tsv(cls, fpath, sep='\t', encoding='utf-8', 
                 index_col=0, header=0, **kwargs):
        """Load expression matrix from a text file.

        Wrapper around `pandas.read_csv()`."""

        matrix = pd.read_csv(
            fpath, encoding=encoding, sep=sep,
            index_col=index_col, header=header, **kwargs)

        matrix = cls(matrix)

        if np.issubdtype(matrix.values.dtype, np.integer):
            matrix = matrix.astype(np.uint32, copy=False)

        fpath_expanded = os.path.expanduser(fpath)
        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            'plain-text format, %.1f MB (hash: %s).',
            matrix.num_cells, matrix.num_genes, file_size_mb, matrix.hash)

        return matrix


    def save_tsv(self, fpath, sep='\t', encoding='utf-8', 
                 float_format='%.5f', **kwargs):
        """Save expression matrix as a plain-text file.

        Wrapper around `pandas.DataFrame.to_csv()`.
        """
        self.to_csv(fpath, sep=sep, encoding=encoding, 
                    float_format=float_format, **kwargs)

        fpath_expanded = os.path.expanduser(fpath)
        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Saved expression matrix with %d cells and %d genes -- '
            'plain-text format, %.1f MB (hash: %s).',
            self.num_cells, self.num_genes, file_size_mb, self.hash)


    @classmethod
    def load_10x_v1(cls, fpath, prefix,
                    use_ensembl_ids=True, use_integer_type=True):
        """Load expression matrix from CellRanger v1/v2 tarball."""

        if not prefix.endswith('/'):
            prefix = prefix + '/'

        with tarfile.open(os.path.expanduser(fpath), mode='r:gz') as tf:
            ti = tf.getmember('%smatrix.mtx' % prefix)
            with tf.extractfile(ti) as fh:
                sparse_data = scipy.io.mmread(fh)
                if use_integer_type:
                    sparse_data = sparse_data.astype(np.uint32)

            ti = tf.getmember('%sgenes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                i = 1
                if use_ensembl_ids:
                    i = 0
                genes = [row[i] for row in csv.reader(wrapper, delimiter='\t')]

            ti = tf.getmember('%sbarcodes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                cells = [row[0] for row in csv.reader(wrapper, delimiter='\t')]

            if sparse_data.shape[0] != len(genes):
                raise ValueError('Number of genes does not match!') 
            if sparse_data.shape[1] != len(cells):
                raise ValueError('Number of cells does not match!') 

        data = sparse_data.todense(order='F')
        matrix = cls(data, genes=genes, cells=cells)

        fpath_expanded = os.path.expanduser(fpath)
        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            'CellRanger v1/v2 sparse format, %.1f MB (hash: %s).',
            matrix.num_cells, matrix.num_genes, file_size_mb, matrix.hash)

        return matrix


    @classmethod
    def load_10x_v2(cls, *args, **kwargs):
        return ExpMatrix.load_10x_v1(*args, **kwargs)


    @classmethod
    def load_10x_v3(cls, fpath, prefix,
                    use_ensembl_ids=True, use_integer_type=True,
                    skip_gene_chars=0):
        """Load expression data from CellRanger v3 tarball."""

        fpath_expanded = os.path.expanduser(fpath)

        if not prefix.endswith('/'):
            prefix = prefix + '/'

        with tarfile.open(fpath_expanded, mode='r:gz') as tf:

            ti = tf.getmember('%smatrix.mtx.gz' % prefix)
            with gzip.open(tf.extractfile(ti)) as fh:
                sparse_data = scipy.io.mmread(fh)
                if use_integer_type:
                    sparse_data = sparse_data.astype(np.uint32)

            ti = tf.getmember('%sfeatures.tsv.gz' % prefix)
            with gzip.open(tf.extractfile(ti)) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                i = 1
                if use_ensembl_ids:
                    i = 0
                genes = [row[i][skip_gene_chars:]
                         for row in csv.reader(wrapper, delimiter='\t')]

            ti = tf.getmember('%sbarcodes.tsv.gz' % prefix)
            with gzip.open(tf.extractfile(ti)) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                cells = [row[0] for row in csv.reader(wrapper, delimiter='\t')]

            if sparse_data.shape[0] != len(genes):
                raise ValueError('Number of genes does not match!') 
            if sparse_data.shape[1] != len(cells):
                raise ValueError('Number of cells does not match!') 

        data = sparse_data.todense(order='F')
        matrix = cls(data, genes=genes, cells=cells)

        file_size_mb = os.path.getsize(fpath_expanded) / 1e6
        _LOGGER.info(
            'Loaded expression matrix with %d cells and %d genes -- '
            'CellRanger v3 sparse format, %.1f MB (hash: %s).',
            matrix.num_cells, matrix.num_genes, file_size_mb, matrix.hash)

        return matrix


    @classmethod
    def load_10x(cls, *args, **kwargs):
        return ExpMatrix.load_10x_v3(cls, *args, **kwargs)


    @classmethod
    def load_starsolo(cls, dir_path: str, use_ensembl_ids: bool = True,
                      trim_ensembl_ids: bool = True,
                      matrix_file_name='matrix.mtx'):
        """Load STARsolo output."""

        barcode_file = os.path.join(dir_path, 'barcodes.tsv')
        gene_file = os.path.join(dir_path, 'features.tsv')
        matrix_file = os.path.join(dir_path, matrix_file_name)

        barcodes = pd.read_csv(
            barcode_file, header=None, sep='\t', squeeze=True)
        genes = pd.read_csv(
            gene_file, header=None, sep='\t')
        if use_ensembl_ids:
            if trim_ensembl_ids:
                genes = genes.iloc[:, 0].str.split('.').apply(lambda x: x[0])
                genes = genes.values.tolist()
            else:
                genes = genes.iloc[:, 0].values.tolist()
        else:
            genes = genes.iloc[:, 1].values.tolist()

        with open(os.path.expanduser(matrix_file), 'rb') as fh:
            mtx = scipy.io.mmread(fh)
        mtx = mtx.astype(np.uint32)

        assert mtx.shape[0] == len(genes)
        assert mtx.shape[1] == len(barcodes)

        _LOGGER.info('Matrix dimensions: %s', str(mtx.shape))
        X = mtx.todense()
        matrix = cls(data=X, genes=genes, cells=barcodes)
        _LOGGER.info('Matrix hash: %s', matrix.hash)
        return matrix


    @classmethod
    def load_enhance(cls, fpath: str):

        from ..denoise import EnhanceModel
        enhance_model = EnhanceModel.load_pickle(fpath)
        matrix = enhance_model.denoised_matrix_
        _LOGGER.info(
            'Obtained denoised expression matrix with %d cells and %d genes '
            'from ENHANCE denoising model (hash: %s).',
            matrix.num_cells, matrix.num_genes, matrix.hash)
        return matrix
