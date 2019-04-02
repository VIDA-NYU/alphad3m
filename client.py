from google.protobuf.timestamp_pb2 import Timestamp
import grpc
import json
import logging
import os
import sys

import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.value_pb2 as pb_value
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
import d3m_ta2_nyu.proto.pipeline_pb2 as pb_pipeline

from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, TASKS_FROM_SCHEMA, \
    SUBTASKS_FROM_SCHEMA
from d3m_ta2_nyu.grpc_logger import LoggingStub


logger = logging.getLogger(__name__)

TASK_TYPES = {n: v for n, v in pb_problem.TaskType.items()}
TASK_SUBTYPES = {n: v for n, v in pb_problem.TaskSubtype.items()}
METRICS = {n: v for n, v in pb_problem.PerformanceMetric.items()}


def do_hello(core):
    core.Hello(pb_core.HelloRequest())


def do_listprimitives(core):
    core.ListPrimitives(pb_core.ListPrimitivesRequest())


def do_search(core, problem):
    version = pb_core.DESCRIPTOR.GetOptions().Extensions[
        pb_core.protocol_version]

    search = core.SearchSolutions(pb_core.SearchSolutionsRequest(
        user_agent='ta3_stub',
        version=version,
        time_bound=2.0,
        allowed_value_types=[pb_value.CSV_URI],
        problem=pb_problem.ProblemDescription(
            problem=pb_problem.Problem(
                id=problem['about']['problemID'],
                version=problem['about']['problemVersion'],
                name=os.path.basename('/input/problem_TRAIN'),
                description="",
                task_type=TASK_TYPES[TASKS_FROM_SCHEMA[
                    problem['about']['taskType']
                ]],
                task_subtype=TASK_SUBTYPES[SUBTASKS_FROM_SCHEMA[
                    problem['about']['taskSubType']
                ]],
                performance_metrics=[
                    pb_problem.ProblemPerformanceMetric(
                        metric=METRICS[SCORES_FROM_SCHEMA[e['metric']]],
                    )
                    for e in problem['inputs']['performanceMetrics']
                ],
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
        ),
        template=pb_pipeline.PipelineDescription(
            id='stub-empty-1',
            source=pb_pipeline.PipelineSource(
                name="NYU stub",
                contact='remi.rampin@nyu.edu',
                pipelines=[],
            ),
            created=Timestamp(seconds=1530545014, nanos=979517000),
            context=pb_pipeline.TESTING,
            name="Stub TA3's empty template",
            description="Empty template",
            users=[pb_pipeline.PipelineDescriptionUser(
                id='stub',
                reason="test run",
                rationale="",
            )],
            inputs=[pb_pipeline.PipelineDescriptionInput(
                name='dataset',
            )],
            outputs=[pb_pipeline.PipelineDescriptionOutput(
                name='dataset',
                data='step.0.produce',
            )],
            steps=[
                pb_pipeline.PipelineDescriptionStep(
                    placeholder=pb_pipeline.PlaceholderPipelineDescriptionStep(
                        inputs=[pb_pipeline.StepInput(data='inputs.0')],
                        outputs=[pb_pipeline.StepOutput(id='produce')],
                    ),
                ),
            ],
        ),
        inputs=[pb_value.Value(
            dataset_uri='file://%s' % '/input/TRAIN/dataset_TRAIN/datasetDoc.json',
        )],
    ))

    results = core.GetSearchSolutionsResults(
        pb_core.GetSearchSolutionsResultsRequest(
            search_id=search.search_id,
        )
    )
    solutions = {}
    for result in results:
        if result.solution_id:
            solutions[result.solution_id] = (
                result.internal_score,
                result.scores,
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


def do_score(core, problem, solutions):
    for solution in solutions:
        try:
            response = core.ScoreSolution(pb_core.ScoreSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % '/input/TRAIN/dataset_TRAIN/datasetDoc.json',
                )],
                performance_metrics=[
                    pb_problem.ProblemPerformanceMetric(
                        metric=METRICS[SCORES_FROM_SCHEMA[e['metric']]],
                    )
                    for e in problem['inputs']['performanceMetrics']
                ],
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


def do_train(core, solutions):
    fitted = {}
    for solution in solutions:
        try:
            response = core.FitSolution(pb_core.FitSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % '/input/TRAIN/dataset_TRAIN/datasetDoc.json',
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


def do_test(core, fitted):
    for fitted_solution in fitted.values():
        try:
            response = core.ProduceSolution(pb_core.ProduceSolutionRequest(
                fitted_solution_id=fitted_solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % '/input/TRAIN/dataset_TRAIN/datasetDoc.json',
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
            for _ in results:
                pass
        except Exception:
            logger.exception("Exception testing %r", fitted_solution)


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

    with open(sys.argv[1]) as problem:
        problem = json.load(problem)

    # Do a hello
    do_hello(core)

    # Do a list primitives
    do_listprimitives(core)
    # Do a search
    solutions = do_search(core, problem)

    # Describe the pipelines
    do_describe(core, solutions)

    # Score all found solutions
    do_score(core, problem, solutions)

    # Train all found solutions
    fitted = do_train(core, solutions)

    # Test all fitted solutions
    do_test(core, fitted)

    # Export all fitted solutions
    do_export(core, fitted)


if __name__ == '__main__':
    main()
