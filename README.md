<img src="https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/raw/devel/AlphaD3M_logo.png" width=30%>


AlphaD3M is an AutoML system that automatically searches for models and derives end-to-end pipelines that read, 
pre-process the data, and train the model. AlphaD3M leverages recent advances in deep reinforcement learning and is 
able to adapt to different application domains and problems through incremental learning.

AlphaD3M provides data scientists and data engineers the flexibility to address complex problems by leveraging the 
Python ecosystem, including open-source libraries and tools, support for collaboration, and infrastructure that enables 
transparency and reproducibility. 

This repository is part of New York University's implementation of the 
[Data Driven Discovery project (D3M)](https://datadrivendiscovery.org/).


Support for Many ML Problems
----------------------------
AlphaD3M uses a comprehensive collection of primitives developed under the D3M program as well as primitives provided 
in open-source libraries, such as scikit-learn, to derive pipelines for a wide range of machine learning tasks. These 
pipelines can be applied to different data types and derive standard performance metrics.

- _Learning Tasks_: classification (semi-supervised, binary, multiclass, and multi-label), regression (univariate, and 
multivariate), time series (forecasting, hierarchical forecasting, and classification),  image-based problems (object 
detection, remote sensing, and image recognition), graph-based problems (collaborative filtering, community detection, 
graph matching, link prediction, and vertex classification),  multi-instance learning and clustering.

- _Data Types_: tabular, time series, hierarchical (grouped, multi-index) time series, geospatial, images, multi-spectral 
imagery, relational, text, graph, audio, video.

- _Data Formats_: D3M, raw CSV, raw text files, OpenML, and scikit-learn datasets.

- _Metrics_: accuracy, F1, macro F1, micro F1, mean squared error, mean absolute error, root mean squared error, object 
detection AP, hamming loss, ROC-AUC, ROC-AUC macro, ROC-AUC micro, jaccard similarity score, normalized mutual 
information, hit at K, R2, recall, mean reciprocal rank, precision, and precision at top K.


Installation
------------
You can use AlphaD3M through [d3m-interface](https://d3m-interface.readthedocs.io/en/latest/).  d3m-interface is a 
Python library to use D3M AutoML systems. This package works with Python 3.6 through 3.8 and  you need to have Docker installed on 
your operating system.

You  can install the latest stable version of this library from [PyPI](https://pypi.org/project/d3m-interface/):

```
$ pip install d3m-interface
```

The first time d3m-interface is used, it automatically downloads a Docker image containing the D3M Core and AlphaD3M.


The documentation of our system can be found [here](https://d3m-interface.readthedocs.io/).
To help users get started with AlphaD3M, we provide Jupyter Notebooks in our
[public repository](https://gitlab.com/ViDA-NYU/d3m/d3m_interface/-/tree/master/examples) that show examples of how the 
library can be used. We also have documentation for the [API](https://d3m-interface.readthedocs.io/en/latest/api.html).


Usability, Model Exploration and Explanation
--------------------------------------------
AlphaD3M greatly simplifies the process to create predictive models. Users can interact with the system from a 
Jupyter notebook, and derive models using a few lines of Python code.

Users can leverage Python-based libraries and tools to clean, transform and visualize data, as well as standard methods 
to explain machine learning models.  They can also be combined to  build customized solutions for specific problems that 
can be deployed to end users.

The AlphaD3M environment includes tools that we developed to enable users to explore the pipelines and their predictions:

- _PipelineProfiler_, an interactive visual analytics tool that empowers data scientists to explore the pipelines derived 
by AlphaD3M within a Jupyter notebook, and gain insights to improve them as well as make an informed decision while 
selecting models for a given application.

- _Visual Text Explorer_, a tool that helps users to understand models for text classification, by allowing to explore 
model predictions and their association with words and entities present in the classified documents.


How AlphaD3M works?
-------------------

Inspired by  AlphaZero, AlphaD3M frames the problem of pipeline synthesis for model discovery as a single-player game 
where the player iteratively builds a pipeline by selecting actions (insertion, deletion and replacement of pipeline 
components). We solve the meta-learning problem using a deep neural network and a Monte Carlo tree search (MCTS). 
The neural network receives as input an entire pipeline, data meta-features, and the problem, and outputs 
action probabilities and estimates for the pipeline performance. The MCTS uses the network probabilities to run 
simulations which terminate at actual pipeline evaluations.
To reduce the search space, we define a pipeline grammar where the rules of the grammar constitute the actions.  The 
grammar rules grow linearly with the number of primitives and hence address the issue of scalability. Finally, AlphaD3M
performs hyperparameter optimization of the best pipelines using SMAC.

For more information about how AlphaD3M works, see our papers:

- [AlphaD3M: Machine Learning Pipeline Synthesis](https://arxiv.org/abs/2111.02508)

- [Automatic Machine Learning by Pipeline Synthesis using Model-Based Reinforcement Learning and a Grammar](https://arxiv.org/abs/1905.10345)
