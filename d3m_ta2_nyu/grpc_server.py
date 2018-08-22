"""GRPC server code, exposing D3mTa2 over the TA3-TA2 protocol.

Those adapters wrap the D3mTa2 object and handle all the GRPC and protobuf
logic, converting to/from protobuf messages. No GRPC or protobuf objects should
leave this module.
"""

import calendar
import datetime
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
import logging
import pickle
from uuid import UUID

from . import __version__

from d3m_ta2_nyu.common import TASKS_FROM_SCHEMA, \
    SCORES_TO_SCHEMA, TASKS_TO_SCHEMA, SUBTASKS_TO_SCHEMA, normalize_score
from d3m_ta2_nyu.grpc_logger import log_service
import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
import d3m_ta2_nyu.proto.value_pb2 as pb_value
import d3m_ta2_nyu.proto.pipeline_pb2 as pb_pipeline
import d3m_ta2_nyu.proto.primitive_pb2 as pb_primitive
from d3m_ta2_nyu.utils import PersistentQueue
import d3m_ta2_nyu.workflow.convert


logger = logging.getLogger(__name__)


def to_timestamp(dt):
    """Converts a UTC datetime object into a gRPC Timestamp.

    :param dt: Time to convert, or None for now.
    :type dt: datetime.datetime | None
    """
    if dt is None:
        dt = datetime.datetime.utcnow()
    return Timestamp(seconds=calendar.timegm(dt.timetuple()),
                     nanos=dt.microsecond * 1000)


def error(context, code, format, *args):
    message = format % args
    context.set_code(code)
    context.set_details(message)
    if code == grpc.StatusCode.NOT_FOUND:
        return KeyError(message)
    else:
        return ValueError(message)


@log_service(logger)
class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = {k: v for v, k in pb_problem.PerformanceMetric.items()
                   if k != pb_problem.METRIC_UNDEFINED}
    metric2grpc = dict(pb_problem.PerformanceMetric.items())

    grpc2task = {k: v for v, k in pb_problem.TaskType.items()
                 if k != pb_problem.TASK_TYPE_UNDEFINED}
    grpc2tasksubtype = {k: v for v, k in pb_problem.TaskSubtype.items()
                        if k != pb_problem.TASK_TYPE_UNDEFINED}

    def __init__(self, ta2):
        self._ta2 = ta2
        self._ta2.add_observer(self._ta2_event)
        self._requests = {}

    def _ta2_event(self, event, **kwargs):
        if 'job_id' in kwargs and kwargs['job_id'] in self._requests:
            job_id = kwargs['job_id']
            self._requests[job_id].put((event, kwargs))
            if event in ('scoring_success', 'scoring_error',
                         'training_success', 'training_error',
                         'test_success', 'test_error'):
                self._requests[job_id].close()

    def SearchSolutions(self, request, context):
        """Create a `Session` and start generating & scoring pipelines.
        """
        if len(request.inputs) > 1:
            raise error(context, grpc.StatusCode.UNIMPLEMENTED,
                        "Search with more than 1 input is not supported")
        expected_version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        if request.version != expected_version:
            logger.error("TA3 is using a different protocol version: %r "
                         "(us: %r)", request.version, expected_version)
        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)
        if not dataset.startswith('file://'):
            dataset = 'file://' + dataset

        problem = self._convert_problem(context, request.problem)

        timeout = request.time_bound
        if timeout < 0.000001:
            timeout = None  # No limit
        else:
            timeout = max(timeout, 1.0)  # At least one minute
            timeout = timeout * 60.0  # Minutes to seconds

        search_id = self._ta2.new_session(problem)

        session = self._ta2.sessions[search_id]
        task = TASKS_FROM_SCHEMA[session.problem['about']['taskType']]
        self._ta2.build_pipelines(search_id,
                                  task,
                                  dataset, session.metrics,
                                  tune=0,  # FIXME: no tuning in TA3 mode
                                  timeout=timeout)

        return pb_core.SearchSolutionsResponse(
            search_id=str(search_id),
        )

    def GetSearchSolutionsResults(self, request, context):
        """Get the created pipelines and scores.
        """
        session_id = UUID(hex=request.search_id)
        if session_id not in self._ta2.sessions:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown search ID %r", session_id)

        session = self._ta2.sessions[session_id]

        def msg_solution(pipeline_id):
            scores = self._ta2.get_pipeline_scores(pipeline_id)

            progress = session.progress

            if scores:
                if session.metrics and session.metrics[0] in scores:
                    metric = session.metrics[0]
                    internal_score = normalize_score(metric, scores[metric],
                                                     'asc')
                else:
                    internal_score = float('nan')
                scores = [
                    pb_core.Score(
                        metric=pb_problem.ProblemPerformanceMetric(
                            metric=self.metric2grpc[m],
                            k=0,
                            pos_label=''),
                        value=pb_value.Value(
                            raw=pb_value.ValueRaw(double=s)
                        ),
                    )
                    for m, s in scores.items()
                    if m in self.metric2grpc
                ]
                scores = [pb_core.SolutionSearchScore(scores=scores)]
                return pb_core.GetSearchSolutionsResultsResponse(
                    done_ticks=progress.current,
                    all_ticks=progress.total,
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Solution scored",
                        start=to_timestamp(session.start),
                    ),
                    solution_id=str(pipeline_id),
                    internal_score=internal_score,
                    scores=scores,
                )

        def msg_progress(status, state=pb_core.RUNNING):
            progress = session.progress

            return pb_core.GetSearchSolutionsResultsResponse(
                done_ticks=progress.current,
                all_ticks=progress.total,
                progress=pb_core.Progress(
                    state=state,
                    status=status,
                    start=to_timestamp(session.start),
                ),
                internal_score=float('nan'),
            )

        with session.with_observer_queue() as queue:
            # Send the solutions that already exist
            for pipeline_id in session.pipelines:
                msg = msg_solution(pipeline_id)
                if msg is not None:
                    yield msg

            # Send updates by listening to notifications on session
            while session.working or not queue.empty():
                if not context.is_active():
                    logger.info(
                        "Client closed GetSearchSolutionsResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session' or event == 'done_searching':
                    break
                elif event == 'new_pipeline':
                    yield msg_progress("Trying new solution")
                elif event == 'scoring_success':
                    pipeline_id = kwargs['pipeline_id']
                    msg = msg_solution(pipeline_id)
                    if msg is not None:
                        yield msg
                    else:
                        yield msg_progress("No appropriate score")
                elif event == 'scoring_error':
                    yield msg_progress("Solution doesn't work")

            yield msg_progress("End of search", pb_core.COMPLETED)

    def EndSearchSolutions(self, request, context):
        """Stop the search and delete the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._ta2.sessions:
            self._ta2.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        """Stop the search without deleting the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._ta2.sessions:
            self._ta2.stop_session(session_id)
            logger.info("Search stopped: %s", session_id)
        return pb_core.StopSearchSolutionsResponse()

    def ScoreSolution(self, request, context):
        """Request scores for a pipeline.
        """
        pipeline_id = UUID(hex=request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        metrics = [self.grpc2metric[m.metric]
                   for m in request.performance_metrics
                   if m.metric in self.grpc2metric]

        logger.info("Got ScoreSolution request, dataset=%s, "
                    "metrics=%s",
                    dataset, metrics)

        pipeline = self._ta2.get_workflow(pipeline_id)
        if pipeline.dataset != dataset:
            # FIXME: Currently scoring only works with dataset in DB
            raise error(context, grpc.StatusCode.UNIMPLEMENTED,
                        "Currently, you can only score on the search dataset")
        # TODO: Get already computed results
        job_id = self._ta2.score_pipeline(pipeline_id, metrics, None)
        self._requests[job_id] = PersistentQueue()

        return pb_core.ScoreSolutionResponse(
            request_id='%x' % job_id,
        )

    def GetScoreSolutionResults(self, request, context):
        """Wait for a scoring job to be done.
        """
        try:
            job_id = int(request.request_id, 16)
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetScoreSolutionResults stream")
                break

            if event == 'scoring_start':
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Scoring in progress",
                    ),
                )
            elif event == 'scoring_success':
                pipeline_id = kwargs['pipeline_id']
                scores = self._ta2.get_pipeline_scores(pipeline_id)
                scores = [
                    pb_core.Score(
                        metric=pb_problem.ProblemPerformanceMetric(
                            metric=self.metric2grpc[m],
                            k=0,
                            pos_label=''),
                        value=pb_value.Value(
                            raw=pb_value.ValueRaw(double=s)
                        ),
                    )
                    for m, s in scores.items()
                    if m in self.metric2grpc
                ]
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Scoring completed",
                    ),
                    scores=scores,
                )
                break
            elif event == 'scoring_error':
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status="Scoring failed",
                    ),
                )
                break

    def FitSolution(self, request, context):
        """Train a pipeline on a dataset.

        This will make it available for testing and exporting.
        """
        pipeline_id = UUID(hex=request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        pipeline = self._ta2.get_workflow(pipeline_id)
        if pipeline.dataset != dataset:
            # FIXME: Currently training only works with dataset in DB
            raise error(context, grpc.StatusCode.UNIMPLEMENTED,
                        "Currently, you can only train on the search dataset")
        job_id = self._ta2.train_pipeline(pipeline_id)
        self._requests[job_id] = PersistentQueue()

        return pb_core.FitSolutionResponse(
            request_id='%x' % job_id,
        )

    def GetFitSolutionResults(self, request, context):
        """Wait for a training job to be done.
        """
        try:
            job_id = int(request.request_id, 16)
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetFitSolutionsResults stream")
                break

            if event == 'training_start':
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Training in progress",
                    ),
                )
            elif event == 'training_success':
                pipeline_id = kwargs['pipeline_id']
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Training completed",
                    ),
                    fitted_solution_id=str(pipeline_id),
                )
                break
            elif event == 'training_error':
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status="Training failed",
                    ),
                )
                break
            elif event == 'done_searching':
                break

    def ProduceSolution(self, request, context):
        """Run testing from a trained pipeline.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        job_id = self._ta2.test_pipeline(pipeline_id, dataset)
        self._requests[job_id] = PersistentQueue()

        return pb_core.ProduceSolutionResponse(
            request_id=job_id,
        )

    def GetProduceSolutionResults(self, request, context):
        """Wait for the requested test run to be done.
        """
        try:
            job_id = request.request_id
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetProduceSolutionResults "
                            "stream")
                break
            if event == 'test_success':
                yield pb_core.GetProduceSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Execution completed",
                    ),
                    exposed_outputs={
                        'outputs.0': pb_value.Value(
                            csv_uri='file://%s' % kwargs['results_path'],
                        ),
                        # FIXME: set 'steps.NN.produce' too in exposed_outputs
                    },
                )
                break
            elif event == 'test_error':
                yield pb_core.GetProduceSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status="Execution failed",
                    ),
                )
                break

    def SolutionExport(self, request, context):
        """Export a trained pipeline as an executable.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._ta2.sessions:
            session = self._ta2.sessions[session_key]
            with session.lock:
                if pipeline_id in session.pipelines:
                    session_id = session.id
                    break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.fitted_solution_id)
        session = self._ta2.sessions[session_id]
        rank = request.rank
        if rank <= 0.0:
            rank = None
        pipeline = self._ta2.get_workflow(pipeline_id)
        if not pipeline.trained:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Solution not fitted: %r",
                        request.fitted_solution_id)
        self._ta2.write_executable(pipeline)
        session.write_exported_pipeline(pipeline_id,
                                        self._ta2.pipelines_exported_root,
                                        rank)
        return pb_core.SolutionExportResponse()

    def Hello(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        user_agent = "nyu_ta2 %s" % __version__

        return pb_core.HelloResponse(
            user_agent=user_agent,
            version=version
        )

    def ListPrimitives(self, request, context):
        raise NotImplementedError

    def DescribeSolution(self, request, context):
        pipeline_id = UUID(hex=request.solution_id)

        pipeline = self._ta2.get_workflow(pipeline_id)
        if not pipeline:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.solution_id)

        steps = []
        step_descriptions = []
        modules = {mod.id: mod for mod in pipeline.modules}
        params = {}
        for param in pipeline.parameters:
            params.setdefault(param.module_id, {})[param.name] = param.value
        module_to_step = {}
        for mod in modules.values():
            self._add_step(steps, step_descriptions, modules, params, module_to_step, mod)

        return pb_core.DescribeSolutionResponse(
            pipeline=pb_pipeline.PipelineDescription(
                id=str(pipeline.id),
                name=str(pipeline.id),
                description=pipeline.origin or '',
                created=to_timestamp(pipeline.created_date),
                context=pb_pipeline.TESTING,
                inputs=[
                    pb_pipeline.PipelineDescriptionInput(
                        name="input dataset"
                    )
                ],
                outputs=[
                    pb_pipeline.PipelineDescriptionOutput(
                        name="predictions",
                        data='steps.%d.produce' % (len(steps) - 1)
                    )
                ],
                steps=steps,
            ),
            steps=step_descriptions
        )

    def _convert_problem(self, context, problem):
        """Convert the problem from the gRPC message to the JSON schema.
        """
        task = problem.problem.task_type
        if task not in self.grpc2task:
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Got unknown task %r", task)
        task = self.grpc2task[task]

        metrics = problem.problem.performance_metrics
        if any(m.metric not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m.metric for m in metrics
                                     if m.metric not in self.grpc2metric))

        metrics = [{'metric': SCORES_TO_SCHEMA[self.grpc2metric[m.metric]]}
                   for m in metrics
                   if m.metric in self.grpc2metric]
        if not metrics:
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Didn't get any metrics we know")

        return {
            'about': {
                'problemID': problem.problem.id,
                'problemVersion': problem.problem.version,
                'problemDescription': problem.problem.description,
                "taskType": TASKS_TO_SCHEMA.get(task, ''),
                "taskSubType": SUBTASKS_TO_SCHEMA.get(
                    self.grpc2tasksubtype.get(problem.problem.task_type),
                    ''),
                "problemSchemaVersion": "3.0",
                "problemName": problem.problem.name,
            },
            'inputs': {
                'performanceMetrics': metrics,
                'data': [
                    {
                        'datasetID': i.dataset_id,
                        'targets': [
                            {
                                'targetIndex': t.target_index,
                                'resID': t.resource_id,
                                'colIndex': t.column_index,
                                'colName': t.column_name,

                            }
                            for t in i.targets
                        ],
                    }
                    for i in problem.inputs
                ],
            },
        }

    def _add_step(self, steps, step_descriptions, modules, params, module_to_step, mod):
        if mod.id in module_to_step:
            return module_to_step[mod.id]

        # Special case: the "dataset" module
        if mod.package == 'data' and mod.name == 'dataset':
            module_to_step[mod.id] = 'inputs.0'
            return 'inputs.0'
        elif mod.package != 'd3m':
            raise ValueError("Got unknown module '%s:%s'" % (mod.package,
                                                             mod.name))

        # Recursively walk upstream modules (to get `steps` in topological
        # order)
        # Add inputs to a dictionary, in deterministic order
        inputs = {}
        for conn in sorted(mod.connections_to, key=lambda c: c.to_input_name):
            step = self._add_step(steps, step_descriptions, modules, params,
                                  module_to_step, modules[conn.from_module_id])
            if step.startswith('inputs.'):
                inputs[conn.to_input_name] = step
            else:
                inputs[conn.to_input_name] = '%s.%s' % (step,
                                                        conn.from_output_name)

        klass = d3m_ta2_nyu.workflow.convert.get_class(mod.name)
        metadata = klass.metadata.query()
        metadata_items = {
            key: metadata[key]
            for key in ('id', 'version', 'python_path', 'name', 'digest')
            if key in metadata
        }

        arguments = {
            name: pb_pipeline.PrimitiveStepArgument(
                container=pb_pipeline.ContainerArgument(
                    data=data,
                )
            )
            for name, data in inputs.items()
        }

        # If hyperparameters are set, export them
        step_hyperparams = {}
        if mod.id in params and 'hyperparams' in params[mod.id]:
            hyperparams = pickle.loads(params[mod.id]['hyperparams'])
            for k, v in hyperparams.items():
                step_hyperparams[k] = pb_pipeline.PrimitiveStepHyperparameter(
                    value=pb_pipeline.ValueArgument(
                        data=pb_value.Value(
                            raw=pb_value.ValueRaw(string=str(v))
                        )
                    )
                )

        # Create step description
        step = pb_pipeline.PipelineDescriptionStep(
            primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                primitive=pb_primitive.Primitive(
                    id=metadata_items['id'],
                    version=metadata_items['version'],
                    python_path=metadata_items['python_path'],
                    name=metadata_items['name'],
                    digest=metadata_items['digest']
                ),
                arguments=arguments,
                outputs=[
                    pb_pipeline.StepOutput(
                        id='produce'
                    )
                ],
                hyperparams=step_hyperparams,
            )
        )

        step_descriptions.append(
            pb_core.StepDescription(
                primitive=pb_core.PrimitiveStepDescription()
            )
        )

        step_nb = 'steps.%d' % len(steps)
        steps.append(step)
        module_to_step[mod.id] = step_nb
        return step_nb
