## CHANGES

### v0.2.2 (2020-06-22)

* Added abilitiy to import/export expression data from/to Scanpy (see new tutorial!).
  * `ExpMatrix.from_anndata()` imports an `AnnData` object (Scanpy)
  * Conversely, `matrix.to_anndata()` exports the expression matrix as an `AnnData` object.

* Added utility function `util.zscore()` to convert a matrix to z-scores.
