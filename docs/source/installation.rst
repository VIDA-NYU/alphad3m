Installation
============

There are two ways to use AlphaD3M: 1) via Docker/Singularity containers (full version), and 2) via PyPI installation
(lightweight version).

Docker/Singularity containers (full version)
---------------------------------------------
AlphaD3M and the primitives will be deployed as a container. This version works with Python 3.6 through 3.8 in Linux,
Mac and Windows. It supports all the ML tasks and data types mentioned :doc:`here <index>`.
You need to have Docker or Singularity installed on your operating system.

To install, run:

::

   $ pip install alphad3m-containers

Once the installation is completed, you need to pull manually the Docker image of AlphaD3M.

For Docker:
::

   $ docker pull registry.gitlab.com/vida-nyu/d3m/alphad3m:latest

or for Singularity:

::

   $ singularity pull docker://registry.gitlab.com/vida-nyu/d3m/alphad3m:latest

PyPI (lightweight version)
---------------------------
Currently, this version has support for classification, regression, time-series classification, time-series forecasting,
semi-supervised classification, collaborative filtering, and clustering (using a limited set of primitives).
It supports tabular, text and image data types. This package works with Python 3.8 in Linux and Mac.

To install, run these commands:

::

   $ pip install alphad3m
   $ pip install d3m-common-primitives d3m-sklearn-wrap dsbox-corex dsbox-primitives sri-d3m distil-primitives rpi-d3m-primitives kf-d3m-primitives d3m-esrnn d3m-nbeats --no-binary pmdarima


The second command installs the primitives available on PyPI.

On non-Linux platforms, you will need `swig` to compile pyrfr. You can obtain swig from `homebrew <https://formulae.brew.sh/formula/swig@3>`__, `anaconda <https://anaconda.org/anaconda/swig>`__, or `the swig website <http://www.swig.org/download.html>`__.
