[![PyPI version](https://badge.fury.io/py/alphad3m.svg)](https://badge.fury.io/py/alphad3m)
[![Pipeline Status](https://gitlab.com/ViDA-NYU/d3m/alphad3m/badges/devel/pipeline.svg)](https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/pipelines/)
[![Documentation Status](https://readthedocs.org/projects/alphad3m/badge/?version=latest)](https://alphad3m.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/alphad3m)](https://pepy.tech/project/alphad3m)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://openreview.net/pdf?id=71eJdMzCCIi)

<img src="https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/raw/devel/AlphaD3M_logo.png" width=30%>


AlphaD3M is an AutoML library that automatically searches for models and derives end-to-end pipelines that read, 
pre-process the data, and train the model. AlphaD3M leverages recent advances in deep reinforcement learning and is 
able to adapt to different application domains and problems through incremental learning.

AlphaD3M provides data scientists and data engineers the flexibility to address complex problems by leveraging the 
Python ecosystem, including open-source libraries and tools, support for collaboration, and infrastructure that enables 
transparency and reproducibility. 

This repository is part of New York University's implementation of the 
[Data Driven Discovery project (D3M)](https://datadrivendiscovery.org/).


**NOTE:**
We recently updated AlphaD3M to use existing open-source primitives that are fully compatible with the scikit-learn API. We call this new version: [Alpha-AutoML](https://github.com/VIDA-NYU/alpha-automl).

## Documentation

Documentation is available [here](https://alphad3m.readthedocs.io/). You can also see how AlphaD3M works in [this video](https://www.youtube.com/watch?v=9qJvOUOh2zM).


## Citing

This is the preferred citation for the AlphaD3M project:

Lopez, Roque, et al. ["AlphaD3M: An Open-Source AutoML Library for Multiple ML Tasks"](https://openreview.net/pdf?id=71eJdMzCCIi), 
AutoML Conference 2023 (ABCD Track), 2023.

Bibtex entry:

```bibtex
@inproceedings{alphad3m,
  title={AlphaD3M: An Open-Source AutoML Library for Multiple ML Tasks},
  author={Lopez, Roque and Lourenco, Raoni and Rampin, Remi and Castelo, Sonia and Santos, Aecio and Ono, Jorge and Silva, Claudio and Freire, Juliana},
  booktitle={AutoML Conference 2023 (ABCD Track)},
  year={2023}
}
```

You can also find [here](https://alphad3m.readthedocs.io/en/latest/how-works.html) our other papers related to the AlphaD3M library. 
