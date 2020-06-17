[![Version][version-shield]][version-url]
[![Python versions][python-shield]][python-url]
[![License][license-shield]][license-url]

![Logo][logo]

# Monet

Monet is an open-source Python package for analyzing and integrating scRNA-Seq data using PCA-based latent spaces.

Datasets from the [Monet paper (Wagner, 2020)](https://www.biorxiv.org/content/10.1101/2020.06.08.140673v2) can be found in a [separate repository](https://github.com/flo-compbio/monet-paper).

Additional documentation is in the works! For questions and requests, please create an "issue" on GitHub.

## Getting started

### Installation

To install Monet, please first use [conda](https://docs.conda.io/en/latest/) to install the packages *pandas*, *scipy*, *scikit-learn*, and *plotly*. If you are new to conda, you can either [install Anaconda](https://docs.anaconda.com/anaconda/install/), which includes all of the aforementioned packages, or you can [install miniconda](https://docs.conda.io/en/latest/miniconda.html) and then manually install these packages. I also recommend using [Jupyter electronic notebooks](https://jupyter.org/) to analyze scRNA-Seq data, which requires installation of the *jupyter* package (also with conda).

Once these prerequisites are installed, you can install Monet using pip:

```sh
$ pip install monet
```

### Tutorials

The following tutorials demonstrate how to use Monet to perform various basic and advanced analsis tasks. The Jupyter electronic notebooks can be [downloaded from GitHub](https://github.com/flo-compbio/monet-tutorials).

#### Basics
1. [Loading and saving expression data](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/010%20-%20Loading%20and%20saving%20expression%20data.ipynb)
2. Importing data from scanpy *(coming soon)*
3. [Visualizing data with t-SNE](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/030%20-%20Visualizing%20data%20with%20t-SNE.ipynb)

#### Clustering
1. [Clustering data with Galapagos (t-SNE plus DBSCAN)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/040%20-%20Clustering%20data%20with%20Galapagos%20%28t-SNE%20plus%20DBSCAN%29.ipynb) *(link currently broken, apologies for the inconvenience)*
2. Annotating clusters with cell types *(coming soon)*

#### Denoising
1. [Denoising data with ENHANCE](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/060%20-%20Denoising%20data%20with%20ENHANCE.ipynb)

#### Data integration
1. [Training a Monet model (for integrative anlayses)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/070%20-%20Train%20a%20Monet%20model%20%28for%20integrative%20analyses%29.ipynb)
2. [Plotting a batch-corrected t-SNE using mutual nearest neighbors (Haghverdi et al.%2C 2018)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/080%20-%20Plot%20a%20batch-corrected%20t-SNE%20using%20mutual%20nearest%20neighbors%20%28Haghverdi%20et%20al.%2C%202018%29.ipynb)
3. [Transferring labels between datasets using K-nearest neighbor classification](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/090%20-%20Label%20transfer%20using%20K-nearest%20neighbor%20classification.ipynb)


## Copyright and License

Copyright (c) 2020 Florian Wagner

Monet is licensed under an OSI-compliant 3-clause BSD license. For details, see [LICENSE](LICENSE).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[version-shield]: https://img.shields.io/pypi/v/monet.svg
[version-url]: https://pypi.python.org/pypi/monet
[python-shield]: https://img.shields.io/pypi/pyversions/monet.svg
[python-url]: https://pypi.python.org/pypi/monet
[license-shield]: https://img.shields.io/pypi/l/monet.svg
[license-url]: https://github.com/flo-compbio/monet/blob/master/LICENSE
[logo]: images/monet_logo_25perc.jpg
