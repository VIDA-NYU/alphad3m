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
import d3m_ta2_vistrails.proto.pipeline_service_pb2 as ps_pb2
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as ps_pb2_grpc


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
        self._pipelines = set()
        self._sessions = {}
        self.executor = None

    def run(self):
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        try:
            server = grpc.server(self.executor)
            ps_pb2_grpc.add_PipelineComputeServicer_to_server(
                self.core_rpc, server)
            for port in self._insecure_ports:
                server.add_insecure_port(port)
            server.start()
            while True:
                time.sleep(60)
        finally:
            self.executor.shutdown(wait=True)

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
        try:
            session_pipelines = self._sessions[session]
        except KeyError:
            session_pipelines = self._sessions[session] = set()
        session_pipelines.add(pipeline_id)

        scores = [
            ps_pb2.Score(
                metric=ps_pb2.ACCURACY,
                value=0.8,
            ),
            ps_pb2.Score(
                metric=ps_pb2.ROC_AUC,
                value=0.5,
            ),
        ]

        return pipeline_id, scores

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


class CoreService(ps_pb2_grpc.PipelineComputeServicer):
    def __init__(self, app):
        self._app = app

    def StartSession(self, request, context):
        version = ps_pb2.DESCRIPTOR.GetOptions().Extensions[
            ps_pb2.protocol_version]
        logger.info("Session started: 1 (protocol version %s)", version)
        return ps_pb2.SessionResponse(
            response_info=ps_pb2.Response(
                status=ps_pb2.Status(code=ps_pb2.OK)
            ),
            user_agent='vistrails_ta2 %s' % __version__,
            version=version,
            context=ps_pb2.SessionContext(session_id='1'),
        )

    def EndSession(self, request, context):
        assert request.session_id == '1'
        logger.info("Session terminated: 1")
        return ps_pb2.Response(
            status=ps_pb2.Status(code=ps_pb2.OK),
        )

    def CreatePipelines(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id == '1'
        train_features = request.train_features
        task = request.task
        assert task == ps_pb2.CLASSIFICATION
        task_subtype = request.task_subtype
        task_description = request.task_description
        output = request.output
        assert output == ps_pb2.FILE
        metrics = request.metrics
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        logger.info("Got CreatePipelines request, session=%s",
                    sessioncontext.session_id)

        results = self._app.build_pipelines(sessioncontext.session_id)
        for pipeline_id, scores in results:
            if not context.is_active():
                logger.info("Client closed CreatePipelines stream")

            yield ps_pb2.PipelineCreateResult(
                response_info=ps_pb2.Response(
                    status=ps_pb2.Status(code=ps_pb2.OK),
                ),
                progress_info=ps_pb2.COMPLETED,
                pipeline_id=pipeline_id,
                pipeline_info=ps_pb2.Pipeline(
                    output=ps_pb2.FILE,
                    scores=scores,
                ),
            )

    def ExecutePipeline(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id == '1'
        pipeline_id = request.pipeline_id

        logger.info("Got ExecutePipeline request, session=%s",
                    sessioncontext.session_id)

        yield ps_pb2.PipelineExecuteResult(
            response_info=ps_pb2.Response(
                status=ps_pb2.Status(code=ps_pb2.OK),
            ),
            progress_info=ps_pb2.COMPLETED,
            pipeline_id=pipeline_id,
        )


def main():
    logging.basicConfig(level=logging.INFO)

    D3mTa2('/tmp/vistrails_ta2').run()
