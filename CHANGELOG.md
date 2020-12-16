Changelog
=========

Version v2021.xx.xx
------------------

Version v2020.12.08
------------------
* Added encoders (text, datetime, etc.) to the grammar.
* Added timeout for pipeline execution (during search). 
* Added support to expouse outputs of the pipeline steps.
* Added support for ROC AUC metric.
* Renamed repository to AlphaD3M

Version v2020.07.24
------------------
* Added support for video data type.
* Improved support for semi-supervised task through SemisupervisedClassificationBuilder class.
* Updated license to Apache-2.0.

Version v2020.06.21
------------------
* Added support for clustering problems.
* Created NN inputs for AlphaD3M from the metalearningDB [#46](https://gitlab.com/ViDA-NYU/d3m/ta2/-/issues/46) [!49](https://gitlab.com/ViDA-NYU/d3m/ta2/-/merge_requests/49)
* Changed the structure of the preprocessing module. Added text and datetime encoders.
* Updated to core package v2020.5.18 and TA2-TA3 API v2020.6.2.

Version v2020.02.16
------------------
Submission for Winter evaluation.

* Added data profiler to the workflow. [#39](https://gitlab.com/ViDA-NYU/d3m/ta2/issues/39) [!47](https://gitlab.com/ViDA-NYU/d3m/ta2/-/merge_requests/47)
* Added support for LUPI problems.
* Added encoders to the search by AlphaD3M.
* Changed versioning to CalVer format.
* Updated to core package v2020.1.9 and TA2-TA3 API v2020.2.11.

1.0 (2019-12-8)
------------------
Submission for December dry-run.

* Added different tasks to the grammar. [#35](https://gitlab.com/ViDA-NYU/d3m/ta2/issues/35) [!42](https://gitlab.com/ViDA-NYU/d3m/ta2/-/merge_requests/42)
* Updated sampling strategy. [#30](https://gitlab.com/ViDA-NYU/d3m/ta2/issues/30) 
[#32](https://gitlab.com/ViDA-NYU/d3m/ta2/issues/32)
* Added templates to external process.
* Updated to core package v2019.11.10 and TA2-TA3 API v2020.12.4.
 

0.9 (2019-06-18)
------------------

Submission for June dry-run.

* Added standard Reference Runtime to execute pipelines
* Added data sampling strategies and priorization of some D3M primitives
* Added `RANK` metric (and corresponding `RANKING` evaluation method) for TA2-only evaluation
* Added `rank_solutions_limit` parameter in `SearchSolutions` which allows request both searching and ranking at the same time
* Updated TA3-TA2 API functions: ListPrimitivesRequest, SearchSolutions and ScoreSolution
* Updated to core package v2019.6.7 and TA2-TA3 API v2019.6.11

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
