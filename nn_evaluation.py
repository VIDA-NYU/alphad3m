"""Easily evaluating pipelines through the TA2 machinery.
"""

import os
import tempfile

from d3m_ta2_nyu.ta2 import D3mTa2
from d3m_ta2_nyu.workflow import database


storage = tempfile.mkdtemp(prefix='d3m_pipeline_eval_')

ta2 = D3mTa2(storage_root=storage,
             logs_root=os.path.join(storage, 'logs'),
             executables_root=os.path.join(storage, 'executables'))


def evaluate_pipeline_from_strings(strings, origin,
                                   dataset, problem):
    """Translate the pipeline, add it to the database, and evaluate it.

    Example::

        evaluate_pipeline_from_strings(
            [
                'dsbox.datapreprocessing.cleaner.KNNImputation',
                'dsbox.datapreprocessing.cleaner.Encoder',
                'sklearn.tree.tree.DecisionTreeClassifier'
            ],
            "Test from command-line",
            'data/LL0_22_mfeat_zernike/LL0_22_mfeat_zernike_dataset',
            'data/LL0_22_mfeat_zernike/LL0_22_mfeat_zernike_problem')

    :param strings: A list of 3 strings [imputer, encoder, classifier]
    :param origin: A description of this pipeline's creator, for example
        "Created by neural network"
    :param dataset: Path to the D3M dataset to use when running the pipeline
    :param problem: Path to the D3M problem to use when running the pipeline
    :return: A dict of scores, for example ``{'F1': 0.5, 'ROC_AUC': 1.0}``
    """

    # Create the pipeline in the database
    db = ta2.DBSession()

    pipeline = database.Pipeline(origin=origin)

    def make_module(package, version, name):
        pipeline_module = database.PipelineModule(
            pipeline=pipeline,
            package=package, version=version, name=name)
        db.add(pipeline_module)
        return pipeline_module

    def connect(from_module, to_module,
                from_output='data', to_input='data'):
        db.add(database.PipelineConnection(from_module=from_module,
                                           to_module=to_module,
                                           from_output_name=from_output,
                                           to_input_name=to_input))

    try:
        data = make_module('data', '0.0', 'data')
        targets = make_module('data', '0.0', 'targets')

        # Assuming a simple linear pipeline
        imputer_name, encoder_name, classifier_name = strings

        # This will use sklearn directly, and others through the TA1 interface
        def make_primitive(name):
            if name.startswith('sklearn.'):
                return make_module('sklearn-builtin', '0.0', name)
            else:
                return make_module('primitives', '0.0', name)

        imputer = make_primitive(imputer_name)
        encoder = make_primitive(encoder_name)
        classifier = make_primitive(classifier_name)

        connect(data, imputer)
        connect(imputer, encoder)
        connect(encoder, classifier)
        connect(targets, classifier, 'targets', 'targets')

        db.add(pipeline)
        db.commit()
        pipeline_id = pipeline.id
    finally:
        db.close()

    # Evaluate the pipeline
    return ta2.run_pipeline(pipeline_id, dataset, problem)
