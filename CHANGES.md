## CHANGES

### v0.2.2 (2020-06-22)

* Added abilitiy to import/export expression data from/to Scanpy (see new tutorial!).
  * `ExpMatrix.from_anndata()` imports an `AnnData` object (Scanpy)
  * Conversely, `matrix.to_anndata()` exports the expression matrix as an `AnnData` object.

* Added utility function `util.zscore()` to convert a matrix to z-scores.

### v0.3.0 (2021-05-04)

* Added support for heatmaps (`workflows.overexpressed_gene_heatmap()`)

* Improved data preprocessing (`preprocess.preprocess_data()`) that allows better reporting of QC metrics

* Support for graph-based clustering using the Leiden algorithm, by running Scanpy's implementation using the Monet latent space model  (`workflows.graph_based_clustering()`)

* Latent space model simplified => Now everything is built around the `PCAModel` class

* Nearest-neighbor aggregation function optimized (now uses scikit-learn `NearestNeighbor` class)

* Support for processing data produced with STARsolo's Velocyto option

* New concepts of **workflows** that provide a simplified API to streamline common analysis tasks (try `from monet import workflows as flow`), including:
  * Data preprocessing (e.g., `workflows.preprocess_10x_v3()`)
  * Visualization (e.g., `workflows.tsne()`)
  * Clustering (e.g., `workflows.graph_based_clustering()`)
