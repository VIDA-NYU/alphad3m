import collections
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
import json
import os
import sys
import time

import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.value_pb2 as pb_value
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
import d3m_ta2_nyu.proto.pipeline_pb2 as pb_pipeline


class LoggingStub(object):
    def __init__(self, stub):
        self._stub = stub

    def __getattr__(self, item):
        return self._wrap(getattr(self._stub, item), item)

    def _wrap(self, method, name):
        def wrapper(out_msg):
            print("< %s" % name)
            for line in str(out_msg).splitlines():
                print("< | %s" % line)
            print("  --------------------")

            start = time.time()
            in_msg = method(out_msg)

            if isinstance(in_msg, collections.Iterable):
                return self._stream(in_msg, start, name)
            else:
                self._print(in_msg, start, name)
                return in_msg

        return wrapper

    def _stream(self, msg, start, name):
        for msg in msg:
            self._print(msg, start, name)
            yield msg

    def _print(self, msg, start, name):
        print("> %s (%f s)" % (name, time.time() - start))
        for line in str(msg).splitlines():
            print("> | %s" % line)
        print("  --------------------")


def main():
    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel))

    version = pb_core.DESCRIPTOR.GetOptions().Extensions[
        pb_core.protocol_version]

    core.Hello(pb_core.HelloRequest())

    with open(sys.argv[1]) as config:
        config = json.load(config)
    with open(config['problem_schema']) as problem:
        problem = json.load(problem)

    TASK_TYPES = {n: v for n, v in pb_problem.TaskType.items()}
    TASK_SUBTYPES = {n: v for n, v in pb_problem.TaskSubtype.items()}
    METRICS = {n: v for n, v in pb_problem.PerformanceMetric.items()}

    solutions = core.SearchSolutions(pb_core.SearchSolutionsRequest(
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
                        metric=METRICS[e['metric'].upper()],
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

    for solution in solutions:
        pass


if __name__ == '__main__':
    main()
