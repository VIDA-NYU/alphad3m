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

from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA
from d3m_ta2_nyu.grpc_logger import LoggingStub


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s")

    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)

    version = pb_core.DESCRIPTOR.GetOptions().Extensions[
        pb_core.protocol_version]

    core.Hello(pb_core.HelloRequest())

    with open(sys.argv[1]) as config:
        config = json.load(config)
    with open(sys.argv[2]) as problem:
        problem = json.load(problem)

    TASK_TYPES = {n: v for n, v in pb_problem.TaskType.items()}
    TASK_SUBTYPES = {n: v for n, v in pb_problem.TaskSubtype.items()}
    METRICS = {n: v for n, v in pb_problem.PerformanceMetric.items()}

    # Do a search
    search = core.SearchSolutions(pb_core.SearchSolutionsRequest(
        user_agent='ta3_stub',
        version=version,
        time_bound=10.0,
        allowed_value_types=[pb_value.CSV_URI],
        problem=pb_problem.ProblemDescription(
            problem=pb_problem.Problem(
                id=problem['about']['problemID'],
                version=problem['about']['problemVersion'],
                name=os.path.basename(config['problem_root']),
                description="",
                task_type=TASK_TYPES[problem['about']['taskType'].upper()],
                task_subtype=TASK_SUBTYPES[problem['about']['taskSubType']
                                           .upper()],
                performance_metrics=[
                    pb_problem.ProblemPerformanceMetric(
                        metric=METRICS[SCORES_FROM_SCHEMA[e['metric']].upper()],
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
            dataset_uri='file://%s' % config['dataset_schema'],
        )],
    ))

    results = core.GetSearchSolutionsResults(pb_core.GetSearchSolutionsResultsRequest(
        search_id=search.search_id
    ))
    solutions = {}
    for result in results:
        if result.solution_id:
            solutions[result.solution_id] = (
                result.internal_score,
                result.scores,
            )

    # Score all found solutions
    for solution in solutions:
        response = core.ScoreSolution(pb_core.ScoreSolutionRequest(
            solution_id=solution,
            inputs=[pb_value.Value(
                dataset_uri='file://%s' % config['dataset_schema'],
            )],
            performance_metrics=[
                pb_problem.ProblemPerformanceMetric(
                    metric=METRICS[SCORES_FROM_SCHEMA[e['metric']].upper()],
                )
                for e in problem['inputs']['performanceMetrics']
            ],
            users=[pb_core.SolutionRunUser(
                id='stub',
                choosen=False,
                reason="test run",
            )],
            configuration=pb_core.ScoringConfiguration(
                method=pb_core.K_FOLD,
                folds=4,
                train_test_ratio=0.75,
                shuffle=True,
            ),
        ))
        results = core.GetScoreSolutionResults(
            pb_core.GetScoreSolutionResultsRequest(
                request_id=response.request_id,
            )
        )
        for _ in results:
            pass


if __name__ == '__main__':
    main()
