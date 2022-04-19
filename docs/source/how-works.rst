How AlphaD3M works
====================

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

- `AlphaD3M: Machine Learning Pipeline Synthesis <https://arxiv.org/abs/2111.02508>`__
- `Automatic Machine Learning by Pipeline Synthesis using Model-Based Reinforcement Learning and a Grammar
  <https://arxiv.org/abs/1905.10345>`__



Support for Many ML Problems
-----------------------------

AlphaD3M uses a comprehensive collection of primitives developed under the D3M program as well as primitives provided
in open-source libraries, such as scikit-learn, to derive pipelines for a wide range of machine learning tasks. These
pipelines can be applied to different data types and derive standard performance metrics.

- *Learning Tasks*: classification, regression, clustering, time series forecasting, time series classification, object
  detection, LUPI, community detection, link prediction, graph matching, vertex classification, collaborative filtering,
  and semi-supervised classification.
- *Data Types*: tabular, text, images, audio, video, and graph.
- *Data Formats*: CSV, D3M, raw text files, OpenML, and scikit-learn datasets.
- *Metrics*: accuracy, F1, macro F1, micro F1, mean squared error, mean absolute error, root mean squared error, object
  detection AP, hamming loss, ROC-AUC, ROC-AUC macro, ROC-AUC micro, jaccard similarity score, normalized mutual
  information, hit at K, R2, recall, mean reciprocal rank, precision, and precision at top K.


Usability, Model Exploration and Explanation
---------------------------------------------

AlphaD3M greatly simplifies the process to create predictive models. Users can interact with the system from a
Jupyter notebook, and derive models using a few lines of Python code.

Users can leverage Python-based libraries and tools to clean, transform and visualize data, as well as standard methods
to explain machine learning models.  They can also be combined to  build customized solutions for specific problems that
can be deployed to end users.

The AlphaD3M environment includes tools that we developed to enable users to explore the pipelines and their predictions:

- *PipelineProfiler*, an interactive visual analytics tool that empowers data scientists to explore the pipelines derived
  by AlphaD3M within a Jupyter notebook, and gain insights to improve them as well as make an informed decision while
  selecting models for a given application.
- *Visual Text Explorer*, a tool that helps users to understand models for text classification, by allowing to explore
  model predictions and their association with words and entities present in the classified documents.


