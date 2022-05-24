Installation
============

There are two ways to use AlphaD3M: 1) via Docker/Singularity containers (full version), and 2) via PyPI installation
(lightweight version).

Docker/Singularity containers (full version)
---------------------------------------------
AlphaD3M and the primitives will be deployed as a container. This version works with Python 3.6 through 3.8 in Linux,
Mac and Windows. It supports all the ML tasks and data types mentioned :doc:`here <how-works>`.
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
Currently, this version has support for classification, regression, clustering, time-series forecasting, time-series
classification, object detection, collaborative filtering, and semi-supervised classification (using a limited set of primitives).
It supports tabular, text, image, audio, and video data types. This package works with Python 3.8 in Linux and Mac.
Installation will require a version of `pip >= 20.3` to leverage the improved dependency resolver, as lower versions may
raise dependency conflicts. You might need GCC or other C/C++ compilers to install packages like NumPy, which uses C
extensions. Also, you will need to have `git` installed in your machine.

To install, run these commands:

::

   $ pip install alphad3m
   $ pip install d3m-common-primitives d3m-sklearn-wrap dsbox-corex dsbox-primitives sri-d3m distil-primitives rpi-d3m-primitives kf-d3m-primitives autonbox fastlvm d3m-esrnn d3m-nbeats --no-binary pmdarima


The second command installs the primitives available on PyPI.

|:warning:| WARNING |:warning:|

On non-Linux platforms:

- You will need `swig` to compile pyrfr. You can obtain swig from
  `homebrew <https://formulae.brew.sh/formula/swig@3>`__, `anaconda <https://anaconda.org/anaconda/swig>`__, or
  `the swig website <http://www.swig.org/download.html>`__.
- To install fastlvm primitives on Mac, run:

  ::

     CFLAGS=-mmacosx-version-min=10.12 CXXFLAGS=-mmacosx-version-min=10.12 python -m pip install fastlvm
