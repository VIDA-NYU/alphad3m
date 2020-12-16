AlphaD3M
=================================

AlphaD3M is an AutoML system that automatically searches for models and derives end-to-end pipelines that read, pre-process the data, and train the model.
AlphaD3M uses deep learning to learn how to incrementally construct these pipelines. The process progresses by self play with iterative self improvement.

This repository is part of New York University's implementation of the `Data Driven Discovery project (D3M) <https://datadrivendiscovery.org/>`__.


Installation
------------


You can use AlphaD3M through `d3m-interface <https://gitlab.com/ViDA-NYU/d3m/d3m_interface>`__.  d3m-interface is a Python library to use D3M AutoML systems.
This package works with Python 3.6 and  you need to have Docker installed on your operating system.

You  can install the latest stable version of this library from `PyPI <https://pypi.org/project/d3m-interface/>`__:

::

    pip3 install d3m-interface


The first time d3m-interface is used, it automatically downloads a Docker image containing the D3M Core and AlphaD3M.


The documentation of our system can be found `here <https://d3m-interface.readthedocs.io/en/latest/getting-started.html>`__.
To help users get started with AlphaD3M, we provide Jupyter Notebooks in our
`public repository <https://gitlab.com/ViDA-NYU/d3m/d3m_interface/-/tree/master/examples>`__ that show examples of how the library can be used.
We also have documentation for the `API <https://d3m-interface.readthedocs.io/en/latest/api.html>`__.
