[![Version][version-shield]][version-url]
[![Python versions][python-shield]][python-url]
[![License][license-shield]][license-url]

![Logo][logo]

# Monet

*Note: This repository contains the scRNA-Seq analysis software. For other tools named Monet, see [Disambiguation](#disambiguation)*

Monet is an open-source Python package for analyzing and integrating scRNA-Seq data using PCA-based latent spaces. Datasets from the [Monet paper (Wagner, 2020)](https://www.biorxiv.org/content/10.1101/2020.06.08.140673v2) can be found in a [separate repository](https://github.com/flo-compbio/monet-paper).

For questions and requests, please create an "issue" on GitHub. For a version history, see [CHANGES](CHANGES.md).

## Getting started

### Installation

The recommended way to install Monet is to first install most of its dependencies using [conda](https://docs.conda.io/en/latest/), and to then install Monet and other dependencies that are not available through conda using [pip](https://pip.pypa.io/en/stable/).

#### 1. Installing Miniconda

If you are new to conda, please [install Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

#### 2. Create a new conda environment for installing Monet

Create a new conda environment named "monet" with Python 3.8 as follows (commands are for Linux/Ubuntu):

```sh
$ conda create -n monet python=3.8
```

#### 3. Use conda to install most of Monet's dependencies

Activate the new environment and install the following packages:

```sh
$ conda activate monet
(monet) $ conda install scikit-learn pandas cython plotly seaborn statsmodels numba pytables networkx click
```

#### 4. Use pip to install the remaining dependencies and Monet itself

Make sure your conda environment is still activated. Then install the following packages:

```sh
(monet) $ pip install leidenalg scanpy monet
```

### Tutorials (v0.2.2)

The following tutorials were developed using Monet v0.2.2. They demonstrate how to use Monet to perform various basic and advanced analysis tasks. The Jupyter electronic notebooks can be [downloaded from GitHub](https://github.com/flo-compbio/monet-tutorials).

#### Basics
1. [Loading and saving expression data](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/010%20-%20Loading%20and%20saving%20expression%20data.ipynb)
2. [Importing/exporting data from/to Scanpy](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/i30w4g0egkhjt5o/020%20-%20Importing%20data%20from%20Scanpy%20and%20exporting%20data%20to%20Scanpy.ipynb)
3. [Visualizing data with t-SNE](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/030%20-%20Visualizing%20data%20with%20t-SNE.ipynb)

#### Clustering
1. [Clustering data with Galapagos (t-SNE + DBSCAN)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/040%20-%20Clustering%20data%20with%20Galapagos%20%28t-SNE%20plus%20DBSCAN%29.ipynb)
2. Annotating clusters with cell types *(coming soon)*

#### Denoising
1. [Denoising data with ENHANCE](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/060%20-%20Denoising%20data%20with%20ENHANCE.ipynb)

#### Data integration
1. [Training a Monet model (for integrative anlayses)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/070%20-%20Train%20a%20Monet%20model%20%28for%20integrative%20analyses%29.ipynb)
2. [Plotting a batch-corrected t-SNE using mutual nearest neighbors (Haghverdi et al.%2C 2018)](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/080%20-%20Plot%20a%20batch-corrected%20t-SNE%20using%20mutual%20nearest%20neighbors%20%28Haghverdi%20et%20al.%2C%202018%29.ipynb)
3. [Transferring labels between datasets using K-nearest neighbor classification](https://nbviewer.jupyter.org/github/flo-compbio/monet-tutorials/blob/master/090%20-%20Label%20transfer%20using%20K-nearest%20neighbor%20classification.ipynb)


## Copyright and License

Copyright (c) 2020-2021 Florian Wagner

Monet is licensed under an OSI-compliant 3-clause BSD license. For details, see [LICENSE](LICENSE).

## Disambiguation

The following other tools have been named Monet (styled either MONET or MONet):

* Overview of the Model and Observation Evaluation Toolkit (MONET) ([Baker and Pan, 2017](https://www.mdpi.com/2073-4433/8/11/210)) \[[github](https://github.com/noaa-oar-arl/MONET)\]
* MONet: Unsupervised Scene Decomposition and Representation ([Burgess et al., 2019](https://arxiv.org/abs/1901.11390)) \[[github](https://github.com/baudm/MONet-pytorch)\]
* MONET: a toolbox integrating top-performing methods for network modularization ([Tomasoni et al., 2020](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa236/5818484)) \[[preprint](https://www.biorxiv.org/content/10.1101/611418v4)\] \[[github](https://github.com/BergmannLab/MONET)\]
* Multi-Objective Cellular Evolutionary Algorithm (MONET) ([García-Nieto et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S1476927118305097)) \[[github](https://github.com/KhaosResearch/monet)\]
* MONET: Multi-omic patient module detection by omic selection ([Rappoport et al., 2020](https://www.biorxiv.org/content/10.1101/2020.02.21.960062v1)) \[[github](https://github.com/Shamir-Lab/MONET)\]

Thanks to Michał Krassowski ([@krassowski\_m](https://twitter.com/krassowski_m)) and Dr. Matthias Stahl ([@h\_i\_g\_s\_c\_h](https://twitter.com/h_i_g_s_c_h)) for providing these references.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[version-shield]: https://img.shields.io/pypi/v/monet.svg
[version-url]: https://pypi.python.org/pypi/monet
[python-shield]: https://img.shields.io/pypi/pyversions/monet.svg
[python-url]: https://pypi.python.org/pypi/monet
[license-shield]: https://img.shields.io/pypi/l/monet.svg
[license-url]: https://github.com/flo-compbio/monet/blob/master/LICENSE
[logo]: images/monet_logo_25perc.jpg
