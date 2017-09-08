from concurrent import futures
import grpc
import logging
import os
import time
import uuid
import vistrails.core.application
from vistrails.core.db.action import create_action
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator, UntitledLocator
from vistrails.core.modules.module_registry import get_module_registry
from vistrails.core.modules.vistrails_module import ModuleError
from vistrails.core.packagemanager import get_package_manager
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails import __version__
import d3m_ta2_vistrails.proto.pipeline_service_pb2 as pb_core
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as pb_core_grpc
import d3m_ta2_vistrails.proto.dataflow_service_pb2 as pb_dataflow
import d3m_ta2_vistrails.proto.dataflow_service_pb2_grpc as pb_dataflow_grpc


logger = logging.getLogger(__name__)


vistrails_app = vistrails.core.application.init(
    options_dict={
        # Don't try to install missing dependencies
        'installBundles': False,
        # Don't enable all packages on start
        'loadPackages': False,
        # Enable packages automatically when they are required
        'enablePackagesSilently': True,
    },
    args=[])


class D3mTa2(object):
    def __init__(self, directory, insecure_ports=None):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, 'workflows')):
            os.mkdir(os.path.join(directory, 'workflows'))
        if insecure_ports is None:
            self._insecure_ports = ['[::]:50051']
        else:
            self._insecure_ports = insecure_ports
        self.core_rpc = CoreService(self)
        self.dataflow_rpc = DataflowService(self)
        self._pipelines = set()
        self._sessions = {}
        self.executor = None

    def run(self):
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        try:
            server = grpc.server(self.executor)
            pb_core_grpc.add_PipelineComputeServicer_to_server(
                self.core_rpc, server)
            pb_dataflow_grpc.add_DataflowServicer_to_server(
                self.dataflow_rpc, server)
            for port in self._insecure_ports:
                server.add_insecure_port(port)
            server.start()
            while True:
                time.sleep(60)
        finally:
            self.executor.shutdown(wait=True)

    def new_session(self):
        session = '%d' % len(self._sessions)
        self._sessions[session] = set()
        return session

    def finish_session(self, session):
        self._sessions.pop(session)

    def has_session(self, session):
        return session in self._sessions

    def build_pipelines(self, session):
        for template in self.TEMPLATES:
            yield self.build_pipeline_from_template(session, template)

    def build_pipeline_from_template(self, session, template):
        pipeline_id = str(uuid.uuid4())

        # Copied from VistrailsApplicationInterface#open_vistrail()
        locator = UntitledLocator()
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])

        # Populate it from a template
        template(self, controller)

        # Save it to disk
        locator = BaseLocator.from_url(os.path.join(self.directory,
                                                    'workflows',
                                                    pipeline_id + '.vt'))
        controller.flush_delayed_actions()
        controller.write_vistrail(locator)

        # Add it to the database
        self._pipelines.add(pipeline_id)
        self._sessions[session].add(pipeline_id)

        scores = [
            pb_core.Score(
                metric=pb_core.ACCURACY,
                value=0.8,
            ),
            pb_core.Score(
                metric=pb_core.ROC_AUC,
                value=0.5,
            ),
        ]

        return pipeline_id, scores

    def get_pipeline(self, session, pipeline_id):
        if pipeline_id not in self._sessions[session]:
            raise KeyError("No such pipeline ID for session")

        filename = os.path.join(self.directory,
                                'workflows',
                                pipeline_id + '.vt')

        # copied from VistrailsApplicationInterface#open_vistrail()
        locator = BaseLocator.from_url(filename)
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    def _example_template(self, controller):
        controller.select_latest_version()
        ops = []
        # Create String module
        mod_string = controller.create_module(
            'org.vistrails.vistrails.basic', 'String')
        ops.append(('add', mod_string))
        # Set the function
        ops.extend(controller.update_function_ops(
            mod_string, 'value', ["Hello, World"]))

        # Create the StandardOutput module
        mod_out = controller.create_module(
            'org.vistrails.vistrails.basic', 'StandardOutput')
        ops.append(('add', mod_out))

        # Add the connection
        connection = controller.create_connection(
            mod_string, 'value',
            mod_out, 'value')
        ops.append(('add', connection))

        action = create_action(ops)
        controller.add_new_action(action)
        version = controller.perform_action(action)
        controller.change_selected_version(version)

    TEMPLATES = [_example_template]


class CoreService(pb_core_grpc.PipelineComputeServicer):
    def __init__(self, app):
        self._app = app

    def StartSession(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        session_id = self._app.new_session()
        logger.info("Session started: %s (protocol version %s)",
                    session_id, version)
        return pb_core.SessionResponse(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK)
            ),
            user_agent='vistrails_ta2 %s' % __version__,
            version=version,
            context=pb_core.SessionContext(session_id=session_id),
        )

    def EndSession(self, request, context):
        self._app.finish_session(request.session_id)
        logger.info("Session terminated: %s", request.session_id)
        return pb_core.Response(
            status=pb_core.Status(code=pb_core.OK),
        )

    def CreatePipelines(self, request, context):
        sessioncontext = request.context
        assert self._app.has_session(sessioncontext.session_id)
        train_features = request.train_features
        task = request.task
        assert task == pb_core.CLASSIFICATION
        task_subtype = request.task_subtype
        task_description = request.task_description
        output = request.output
        assert output == pb_core.FILE
        metrics = request.metrics
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        logger.info("Got CreatePipelines request, session=%s",
                    sessioncontext.session_id)

        results = self._app.build_pipelines(sessioncontext.session_id)
        for pipeline_id, scores in results:
            if not context.is_active():
                logger.info("Client closed CreatePipelines stream")

            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.OK),
                ),
                progress_info=pb_core.COMPLETED,
                pipeline_id=pipeline_id,
                pipeline_info=pb_core.Pipeline(
                    output=pb_core.FILE,
                    scores=scores,
                ),
            )

    def ExecutePipeline(self, request, context):
        sessioncontext = request.context
        assert self._app.has_session(sessioncontext.session_id)
        pipeline_id = request.pipeline_id

        logger.info("Got ExecutePipeline request, session=%s",
                    sessioncontext.session_id)

        # TODO: ExecutePipeline
        yield pb_core.PipelineExecuteResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            progress_info=pb_core.COMPLETED,
            pipeline_id=pipeline_id,
        )


class DataflowService(pb_dataflow_grpc.DataflowServicer):
    def __init__(self, app):
        self._app = app

    def DescribeDataflow(self, request, context):
        sessioncontext = request.context
        assert self._app.has_session(sessioncontext.session_id)
        pipeline_id = request.pipeline_id

        # Build description from VisTrails workflow
        controller = self._app.get_pipeline(sessioncontext.session_id,
                                            pipeline_id)
        vt_pipeline = controller.current_pipeline
        modules = []
        for vt_module in vt_pipeline.module_list:
            functions = dict((func.name, func.params[0].strValue)
                             for func in vt_module.functions
                             if len(func.param) == 1)
            inputs = []
            for port in vt_module.destinationPorts():
                port = pb_dataflow.DataflowDescription.Input(
                    name=port.name,
                    type=port.sigstring,
                )
                if port.name in functions:
                    port.value.CopyFrom(functions[port.name])
                inputs.append(port)
            outputs = [
                pb_dataflow.DataflowDescription.Output(
                    name=port.name,
                    type=port.sigstring,
                )
                for port in vt_module.sourcePorts()
            ]

            modules.append(pb_dataflow.DataflowDescription.Module(
                id='%d' % vt_module.id,
                type=vt_module.module_descriptor.sigstring,
                label=vt_module.module_descriptor.name,
                inputs=inputs,
                outputs=outputs,
            ))

        connections = []
        for vt_connection in vt_pipeline.connection_list:
            connections.append(pb_dataflow.DataflowDescription.Connection(
                from_module_id='%d' % vt_connection.source.moduleId,
                from_output_name=vt_connection.source.name,
                to_module_id='%d' % vt_connection.destination.moduleId,
                to_input_name=vt_connection.destination.name,
            ))

        return pb_dataflow.DataflowDescription(
            pipeline_id=pipeline_id,
            modules=modules,
            connections=connections,
        )

    def GetDataflowResults(self, request, context):
        sessioncontext = request.context
        assert self._app.has_session(sessioncontext.session_id)
        pipeline_id = request.pipeline_id

        # TODO: GetDataflowResults
        yield pb_dataflow.ModuleResult(
            module_id='module1',
            status=pb_dataflow.ModuleResult.RUNNING,
            progress=0.5,
        )
        yield pb_dataflow.ModuleResult(
            module_id='module1',
            status=pb_dataflow.ModuleResult.DONE,
            progress=1.0,
            outputs=[
                pb_dataflow.ModuleOutput(
                    output_name='result',
                    value='42',
                ),
            ],
        )


def main():
    logging.basicConfig(level=logging.INFO)

    D3mTa2('/tmp/vistrails_ta2').run()
