import logging
import numpy
import d3m_ta2_nyu.proto.core_pb2 as pb_core
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.pipeline_evaluation import cross_validation, holdout
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.common import normalize_score, SCORES_FROM_SCHEMA

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 100
RANDOM_SEED = 65682867


@database.with_db
def score(pipeline_id, dataset, metrics, problem, scoring_conf, msg_queue, db):
    if scoring_conf is None:
        scoring_conf = {'train_test_ratio': '0.75',
                        'shuffle': 'true',
                        'stratified': 'false'
        }
        evaluation_method = None
    else:
        evaluation_method = scoring_conf['method']

    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to score pipeline, id=%s, metrics=%s, dataset=%r', pipeline_id, metrics, dataset)

    # Load data

    dataset = Dataset.load(dataset)

    logger.info('Loaded dataset')

    if evaluation_method is None:  # It comes from search_solutions, so do the sample and use HOLDOUT
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
        else:
            res_id = next(iter(dataset))
        if hasattr(dataset[res_id], 'columns') and len(dataset[res_id]) > SAMPLE_SIZE:
            # Sample the dataset to stay reasonably fast
            logger.info('Sampling down data from %d to %d', len(dataset[res_id]), SAMPLE_SIZE)
            sample = numpy.concatenate([numpy.repeat(True, SAMPLE_SIZE),
                                        numpy.repeat(False, len(dataset[res_id]) - SAMPLE_SIZE)])

            numpy.random.RandomState(seed=RANDOM_SEED).shuffle(sample)
            dataset[res_id] = dataset[res_id][sample]

        evaluation_method = pb_core.EvaluationMethod.Value('HOLDOUT')

    if evaluation_method == pb_core.EvaluationMethod.Value('K_FOLD'):
        scores = cross_validation(pipeline, dataset, metrics, problem, scoring_conf)

        # Store scores
        cross_validation_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                cross_validation_scores.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

        crossval_db = database.CrossValidation(pipeline_id=pipeline_id, scores=cross_validation_scores)
        db.add(crossval_db)

    elif evaluation_method == pb_core.EvaluationMethod.Value('HOLDOUT'):
        scores = holdout(pipeline, dataset, metrics, problem, scoring_conf)

        # Store scores
        holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                holdout_scores.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

        # TODO Create Holdout table in database
        holdout_db = database.CrossValidation(pipeline_id=pipeline_id, scores=holdout_scores)
        db.add(holdout_db)

    elif evaluation_method == pb_core.EvaluationMethod.Value('RANKING'):
        metrics = []
        for metric in problem['inputs']['performanceMetrics']:
            metric_name = metric['metric']
            try:
                metric_name = SCORES_FROM_SCHEMA[metric_name]
            except KeyError:
                logger.error("Unknown metric %r", metric_name)
                raise ValueError("Unknown metric %r" % metric_name)

            formatted_metric = {'metric': metric_name}

            if len(metric) > 1:  # Metric has parameters
                formatted_metric['params'] = {}
                for param in metric.keys():
                    if param != 'metric':
                        formatted_metric['params'][param] = metric[param]

            metrics.append(formatted_metric)

        scores = holdout(pipeline, dataset, metrics, problem, {})

        # Store scores
        holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, current_score in fold_scores.items():
                new_score = normalize_score(metric, current_score, 'desc')
                holdout_scores.append(database.CrossValidationScore(fold=fold, metric='RANK', value=new_score))

        # TODO Create Ranking table in database
        holdout_db = database.CrossValidation(pipeline_id=pipeline_id, scores=holdout_scores)
        db.add(holdout_db)

    db.commit()
