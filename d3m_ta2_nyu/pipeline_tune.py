import logging
import os
import sys
import shutil
import pickle
import d3m_ta2_nyu.grpc_api.core_pb2 as pb_core
from d3m import index
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.pipeline_score import evaluate, kfold_tabular_split, score
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.parameter_tuning.primitive_config import is_tunable
from d3m_ta2_nyu.parameter_tuning.bayesian import HyperparameterTuning, hyperparams_from_config
from d3m.metadata.problem import TaskKeyword

logger = logging.getLogger(__name__)

PRIMITIVES = index.search()


@database.with_db
def tune(pipeline_id, metrics, problem, dataset_uri, sample_dataset_uri, do_rank, timeout, targets, msg_queue, db):
    timeout = timeout * 0.9  # FIXME: Save 10% of timeout to score the best config
    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to tune pipeline, id=%s, dataset=%r, timeout=%d secs', pipeline_id, dataset_uri, timeout)
    tunable_primitives = {}

    for primitive in pipeline.modules:
        if is_tunable(primitive.name):
            tunable_primitives[primitive.id] = primitive.name

    if len(tunable_primitives) == 0:
        logger.info('No primitives to be tuned for pipeline %s', pipeline_id)
        sys.exit(1)

    logger.info('Tuning primitives: %s', ', '.join(tunable_primitives.values()))

    if sample_dataset_uri:
        dataset = Dataset.load(sample_dataset_uri)
    else:
        dataset = Dataset.load(dataset_uri)
    tuning = HyperparameterTuning(tunable_primitives.values())
    task_keywords = problem['problem']['task_keywords']

    scoring_config = {'shuffle': 'true',
                      'stratified': 'true' if TaskKeyword.CLASSIFICATION in task_keywords else 'false',
                      'method': pb_core.EvaluationMethod.Value('K_FOLD'),
                      'number_of_folds': '2'}

    def evaluate_tune(hyperparameter_configuration):
        new_hyperparams =[]
        for primitive_id, primitive_name in tunable_primitives.items():
            hy = hyperparams_from_config(primitive_name, hyperparameter_configuration)
            db_hyperparams = database.PipelineParameter(
                pipeline=pipeline,
                module_id=primitive_id,
                name='hyperparams',
                value=pickle.dumps(hy),
            )
            new_hyperparams.append(db_hyperparams)

        pipeline.parameters += new_hyperparams
        scores = evaluate(pipeline, kfold_tabular_split, dataset, metrics, problem, scoring_config)
        first_metric = metrics[0]['metric'].name
        score_values = []
        for fold_scores in scores.values():
            for metric, score in fold_scores.items():
                if metric == first_metric:
                    score_values.append(score)

        avg_score = sum(score_values) / len(score_values)
        cost = 1.0 - metrics[0]['metric'].normalize(avg_score)
        logger.info('Tuning results:\n%s, cost=%s', scores, cost)

        return cost

    # Run tuning, gets best configuration
    best_configuration = tuning.tune(evaluate_tune, wallclock=timeout)

    # Duplicate pipeline in database
    new_pipeline = database.duplicate_pipeline(db, pipeline, 'Hyperparameter tuning from pipeline %s' % pipeline_id)

    for primitive_id, primitive_name in tunable_primitives.items():
        best_hy = hyperparams_from_config(primitive_name, best_configuration)

        db.add(database.PipelineParameter(
            pipeline=new_pipeline,
            module_id=primitive_id,
            name='hyperparams',
            value=pickle.dumps(best_hy),
        ))
    db.commit()

    logger.info('Tuning done, generated new pipeline %s', new_pipeline.id)

    for f in os.listdir('/tmp'):
        if 'run_1' in f:
            shutil.rmtree(os.path.join('/tmp', f))

    score(new_pipeline.id, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config, do_rank, None,
          db_filename=os.path.join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'db.sqlite3'))
    # TODO: Change this static string path

    msg_queue.send(('tuned_pipeline_id', new_pipeline.id))
