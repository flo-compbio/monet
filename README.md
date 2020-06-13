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
