import grpc
import json
import logging
import os
import sys
import datetime
import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.value_pb2 as pb_value
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, TASKS_FROM_SCHEMA, SUBTASKS_FROM_SCHEMA
from d3m_ta2_nyu.grpc_logger import LoggingStub


logger = logging.getLogger(__name__)

TASK_TYPES = {n: v for n, v in pb_problem.TaskType.items()}
TASK_SUBTYPES = {n: v for n, v in pb_problem.TaskSubtype.items()}
METRICS = {n: v for n, v in pb_problem.PerformanceMetric.items()}


def do_hello(core):
    core.Hello(pb_core.HelloRequest())


def do_listprimitives(core):
    core.ListPrimitives(pb_core.ListPrimitivesRequest())


def do_search(core, problem, dataset_path, time_bound=30.0, pipelines_limit=0, template=None):
    version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]

    metrics = []

    for m in problem['inputs']['performanceMetrics']:
        if 'posLabel' in m:
            metrics.append(pb_problem.ProblemPerformanceMetric(
                metric=METRICS[SCORES_FROM_SCHEMA[m['metric']]],
                pos_label=m['posLabel'])
            )
        else:
            metrics.append(pb_problem.ProblemPerformanceMetric(
                metric=METRICS[SCORES_FROM_SCHEMA[m['metric']]],)
            )

    search = core.SearchSolutions(pb_core.SearchSolutionsRequest(
        user_agent='ta3_stub',
        version=version,
        time_bound_search=time_bound,
        rank_solutions_limit=pipelines_limit,
        allowed_value_types=[pb_value.CSV_URI],
        problem=pb_problem.ProblemDescription(
            id=problem['about']['problemID'],
            version=problem['about']['problemVersion'] if 'problemVersion' in problem['about'] else '0.0.1',
            name=os.path.basename('/input/problem_TRAIN'),
            description="",
            problem=pb_problem.Problem(
                task_type=TASK_TYPES[TASKS_FROM_SCHEMA[
                    problem['about']['taskType']
                ]],
                task_subtype=TASK_SUBTYPES[SUBTASKS_FROM_SCHEMA[
                    problem['about'].get('taskSubType', 'none')
                ]],
                performance_metrics=metrics
            ),
            inputs=[
                pb_problem.ProblemInput(
                    dataset_id=i['datasetID'],
                    targets=[
                        pb_problem.ProblemTarget(
                            target_index=t['targetIndex'],
                            resource_id=t['resID'],
                            column_index=t['colIndex'],
                            column_name=t['colName'],
                        )
                        for t in i['targets']
                    ],
                )
                for i in problem['inputs']['data']
            ],
            data_augmentation=[
                pb_problem.DataAugmentation(
                    domain=i.get('domain', []),
                    keywords=i.get('keywords', []),
                )
                for i in problem.get('dataAugmentation', []) if i.get('domain', []) or i.get('keywords', [])
            ],
        ),
        inputs=[pb_value.Value(
            dataset_uri='file://%s' % dataset_path,
        )],
    ))

    start_time = datetime.datetime.now()
    results = core.GetSearchSolutionsResults(
        pb_core.GetSearchSolutionsResultsRequest(
            search_id=search.search_id,
        )
    )
    solutions = {}
    for result in results:
        if result.solution_id:
            end_time = datetime.datetime.now()
            solutions[result.solution_id] = (
                result.internal_score,
                result.scores,
                str(end_time - start_time)
            )

    return solutions


def do_describe(core, solutions):
    for solution in solutions:
        try:
            core.DescribeSolution(pb_core.DescribeSolutionRequest(
                solution_id=solution,
            ))
        except Exception:
            logger.exception("Exception during describe %r", solution)


def do_score(core, problem, solutions, dataset_path):
    metrics = []

    for m in problem['inputs']['performanceMetrics']:
        if 'posLabel' in m:
            metrics.append(pb_problem.ProblemPerformanceMetric(
                metric=METRICS[SCORES_FROM_SCHEMA[m['metric']]],
                pos_label=m['posLabel'])
            )
        else:
            metrics.append(pb_problem.ProblemPerformanceMetric(
                metric=METRICS[SCORES_FROM_SCHEMA[m['metric']]], )
            )

    for solution in solutions:
        try:
            response = core.ScoreSolution(pb_core.ScoreSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                performance_metrics=metrics,
                users=[],
                configuration=pb_core.ScoringConfiguration(
                    method=pb_core.EvaluationMethod.Value('K_FOLD'),
                    folds=4,
                    train_test_ratio=0.75,
                    shuffle=True,
                    random_seed=42
                ),
            ))
            results = core.GetScoreSolutionResults(
                pb_core.GetScoreSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for _ in results:
                pass
        except Exception:
            logger.exception("Exception during scoring %r", solution)


def do_train(core, solutions, dataset_path):
    fitted = {}
    for solution in solutions:
        try:
            response = core.FitSolution(pb_core.FitSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=[],
                expose_value_types=[pb_value.CSV_URI],
                users=[],
            ))
            results = core.GetFitSolutionResults(
                pb_core.GetFitSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    fitted[solution] = result.fitted_solution_id
        except Exception:
            logger.exception("Exception training %r", solution)
    return fitted


def do_test(core, fitted, dataset_path):
    tested = {}
    for fitted_solution in fitted.values():
        try:
            response = core.ProduceSolution(pb_core.ProduceSolutionRequest(
                fitted_solution_id=fitted_solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=[],
                expose_value_types=[pb_value.CSV_URI],
                users=[],
            ))
            results = core.GetProduceSolutionResults(
                pb_core.GetProduceSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    tested[fitted_solution] = result.exposed_outputs['outputs.0'].csv_uri
        except Exception:
            logger.exception("Exception testing %r", fitted_solution)

    return tested


def do_export(core, fitted):
    for i, fitted_solution in enumerate(fitted.values()):
        try:
            core.SolutionExport(pb_core.SolutionExportRequest(
                solution_id=fitted_solution,
                rank=(i + 1.0) / (len(fitted) + 1.0),
            ))
        except Exception:
            logger.exception("Exception exporting %r", fitted_solution)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s")

    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)
    train_dataset_path = '/input/TRAIN/dataset_TRAIN/datasetDoc.json'
    test_dataset_path = '/input/TEST/dataset_TEST/datasetDoc.json'

    with open(sys.argv[1]) as problem:
        problem = json.load(problem)

    # Do a hello
    do_hello(core)

    # Do a list primitives
    do_listprimitives(core)
    # Do a search
    solutions = do_search(core, problem, train_dataset_path)

    # Describe the pipelines
    do_describe(core, solutions)

    # Score all found solutions
    do_score(core, problem, solutions, train_dataset_path)

    # Train all found solutions
    fitted = do_train(core, solutions, train_dataset_path)

    # Test all fitted solutions
    do_test(core, fitted, test_dataset_path)

    # Export all fitted solutions
    do_export(core, fitted)


if __name__ == '__main__':
    main()
