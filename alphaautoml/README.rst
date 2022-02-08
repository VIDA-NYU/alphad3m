AlphaAutoML - Automated generation of data mining pipelines
===========================================================

This repository contains the implementation of the AlphaAutoML framework that generates data mining pipelines using machine learning algorithms. A data mining pipeline typically consists of data cleaning, data preprocessing, feature extraction, feature selection and estimator primitives.

This framework is generic and allows generation of pipelines using any given set of primitives and an execution engine that can execute a pipeline generated from the given set of primitives.

Installation and other instructions to follow shortly...

Usage Example
-------------

The script `SklearnPipelineGenerator.py <https://gitlab.com/ViDA-NYU/alphaautoml/blob/master/alphaAutoMLEdit/SklearnPipelineGenerator.py>`__ shows how to use the AlphAutoML framework for sklearn primitives using datasets in the format used for D3M `here <https://gitlab.datadrivendiscovery.org/d3m/datasets/tree/master/training_datasets/>`__.

To run the example you need to install::
  `BYU's metalearn <https://github.com/byu-dml/metalearn>`__. Follow the setup instructions.
  
  scikit-learn==0.20.0
  
It can be run with the following command in the root directory::


  python -m alphaAutoMLEdit.SklearnPipelineGenerator <dataset_name> <dataset_path> <output_path>


NOTE: This script needs to be modified to suit your datasets and primitives.


