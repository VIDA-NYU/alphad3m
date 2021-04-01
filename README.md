<img src="https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/raw/devel/AlphaD3M_logo.png" width=30%>


AlphaD3M is an AutoML system that automatically searches for models and derives end-to-end pipelines that read, 
pre-process the data, and train the model. AlphaD3M leverages recent advances in deep reinforcement learning and is 
able to adapt to different application domains and problems through incremental learning.

This repository is part of New York University's implementation of the 
[Data Driven Discovery project (D3M)](https://datadrivendiscovery.org/).


Installation
------------


You can use AlphaD3M through [d3m-interface](https://d3m-interface.readthedocs.io/en/latest/).  d3m-interface is a 
Python library to use D3M AutoML systems. This package works with Python 3.6 and  you need to have Docker installed on 
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

- [AlphaD3M: Machine Learning Pipeline Synthesis](https://cims.nyu.edu/~drori/alphad3m-paper.pdf)

- [Automatic Machine Learning by Pipeline Synthesis using Model-Based Reinforcement Learning and a Grammar](https://arxiv.org/abs/1905.10345)