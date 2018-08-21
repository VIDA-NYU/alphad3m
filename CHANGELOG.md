Changelog
=========

0.8.1 (2018-08-08)
------------------

Re-submission for 2018 Summer evaluation after Gov team mixup on TA1 library freeze

* Finish implementing gRPC server (which we were initially planning to do before TA2 and TA3 deadlines)
* Use correct base image, mandated by Gov
* Added an 8-minute timeout to `ScoreJob` (some primitives freeze)
* Only report scored pipelines to TA3, don't inform them of created-not-yet-scored (or broken) pipelines

0.8 (2018-08-01)
----------------

First submission for 2018 Summer evaluation (original deadline).

* Build from common `jpl/docker_images/complete` images
* Use `d3m` package to load dataset, remove MIT-LL's `d3mds.py`
* Added `eval.sh` entrypoint to support Data Machine's eval protocol
* Updated gRPC to v2018.7.7
* Training/testing is now independent of `Session`, which only handles searching
* Add a timeout on AlphaD3M
* Do tuning after top pipelines have been trained and written out (then train the tuned pipelines)
* Use `KFoldDatasetSplit` primitive to do cross-validation splits

0.7 (2018-06-06)
----------------

* Added AlphaD3M pipeline generation
* Enabled huperparameter tuning with SMAC
* Added the `Job` class for the run queue
* Introduced own multiprocessing code using sockets and avoiding fork issues

0.6.2 (2018-03-13)
------------------

Bug fixes.

0.6.1 (2018-02-08)
------------------

Bug fixes.

0.6 (2018-02-07)
----------------

Bug fixes.

0.5 (2018-01-31)
----------------

Version submitted to NIST for 2018 January evaluation.

* Added hyperparameter tuning with SMAC. **Disabled**, does not work
* Raise the number of pipelines by using one of 3 imputers, one of 2 encoder
* Updated gRPC to v2017.12.20

0.4 (2018-01-16)
----------------

January dry-run version.

* Removed VisTrails
* Moved from Python 2.7 to Python 3.6
* Renamed package from `d3m_ta2_vistrails` to `d3m_ta2_nyu`
* Use `d3mds.py` from MIT-LL to load dataset
* Use some D3M primitives, in addition to "native" scikit-learn: `KNNImputation` and `Encoder` from ISI's dsbox

0.3 (2017-12-07)
----------------

* Added CI
* Updated gRPC to v2017.10.10

0.2 (2017-10-05)
----------------

Version submitted to NIST for 2017 Fall TA3 evaluation.

* Improvement to data-reading code
* Create directories

0.1 (2017-10-02)
----------------

Version submitted to NIST for 2017 Fall TA2 evaluation.

* Using gRPC protocol v2017.9.11
* Custom data-reading code, identifies column types, does PCA for image data

0.0 (2017-08-24)
----------------

Start of project, using VisTrails for workflow representation and execution.
