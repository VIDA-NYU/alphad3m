from concurrent import futures
import functools
import grpc
import logging
import os
import threading
import time
import uuid
import vistrails.core.application
from vistrails.core.db.action import create_action
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator, UntitledLocator
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails import __version__
import d3m_ta2_vistrails.proto.core_pb2 as pb_core
import d3m_ta2_vistrails.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_vistrails.proto.dataflow_ext_pb2 as pb_dataflow
import d3m_ta2_vistrails.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc


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


vistrails_lock = threading.RLock()


def with_vistrails_lock(wrapped):
    def wrapper(*args, **kwargs):
        with vistrails_lock:
            return wrapped(*args, **kwargs)
    functools.update_wrapper(wrapper, wrapped)
    return wrapper


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
            pb_core_grpc.add_CoreServicer_to_server(
                self.core_rpc, server)
            pb_dataflow_grpc.add_DataflowExtServicer_to_server(
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

    def get_pipelines(self, session):
        return self._sessions[session]

    def build_pipelines(self, session):
        for template_iter in self.TEMPLATES:
            for template in template_iter:
                if isinstance(template, (list, tuple)):
                    func, args = template[0], template[1:]
                    tpl_func = lambda s: func(s, *args)
                else:
                    tpl_func = template
                try:
                    ret = self.build_pipeline_from_template(session, tpl_func)
                except Exception:
                    logger.exception("Error building pipeline from %r",
                                     template)
                else:
                    yield ret

    @with_vistrails_lock
    def build_pipeline_from_template(self, session, template):
        pipeline_id = str(uuid.uuid4())

        # Create workflow from a template
        controller = template(self)

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

    def _new_controller(self):
        # Copied from VistrailsApplicationInterface#open_vistrail()
        locator = UntitledLocator()
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    def _load_template(self, name):
        # Copied from VistrailsApplicationInterface#open_vistrail()
        locator = BaseLocator.from_url(
            os.path.join(os.path.dirname(__file__),
                         '..',
                         'pipelines',
                         name))
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    @staticmethod
    def _replace_module(controller, ops, old_module_id, new_module):
        ops.append(('add', new_module))
        pipeline = controller.current_pipeline
        up_list = pipeline.graph.inverse_adjacency_list[old_module_id]
        for up_mod_id, up_conn_id in up_list:
            up_conn = pipeline.connections[up_conn_id]
            # Remove old connection
            ops.append(('delete', up_conn))
            # Add new connection
            new_up_conn = controller.create_connection(
                pipeline.modules[up_conn.source.moduleId], up_conn.source.name,
                new_module, up_conn.destination.name)
            ops.append(('add', new_up_conn))

        down_list = pipeline.graph.adjacency_list[old_module_id]
        assert len(down_list) <= 1
        for down_mod_id, down_conn_id in down_list:
            down_conn = pipeline.connections[down_conn_id]
            # Remove old connection
            ops.append(('delete', down_conn))
            # Add new connection
            new_down_conn = controller.create_connection(
                new_module, down_conn.source.name,
                pipeline.modules[down_conn.destination.moduleId],
                down_conn.destination.name)
            ops.append(('add', new_down_conn))

        ops.append(('delete', pipeline.modules[old_module_id]))

    def _example_template(self):
        controller = self._new_controller()
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
        return controller

    @staticmethod
    def _get_module(pipeline, label):
        for module in pipeline.module_list:
            if '__desc__' in module.db_annotations_key_index:
                name = module.get_annotation_by_key('__desc__').value
                if name == label:
                    return module
        return None

    def _classification_template(self, classifier_name):
        controller = self._load_template('classification.xml')
        ops = []

        # Replace the classifier module
        module = self._get_module(controller.current_pipeline, 'Classifier')
        if module is not None:
            new_module = controller.create_module(
                'org.vistrails.vistrails.sklearn',
                classifier_name,
                namespace='classifiers')
            self._replace_module(controller, ops,
                                 module.id, new_module)
        else:
            raise ValueError("Couldn't find Classifier module in "
                             "classification template")

        # Set the correct input files
        module = self._get_module(controller.current_pipeline, 'Train')
        if module is not None:
            ops.extend(controller.update_function_ops(
                module, 'name', ['TODO']))
        module = self._get_module(controller.current_pipeline, 'Target')
        if module is not None:
            ops.extend(controller.update_function_ops(
                module, 'name', ['TODO']))

        action = create_action(ops)
        controller.add_new_action(action)
        version = controller.perform_action(action)
        controller.change_selected_version(version)
        return controller

    TEMPLATES = [
        [_example_template],
        [(_classification_template, 'LinearSVC'),
         (_classification_template, 'KNeighborsClassifier'),
         (_classification_template, 'DecisionTreeClassifier')]
    ]


class CoreService(pb_core_grpc.CoreServicer):
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

    def ListPipelines(self, request, context):
        sessioncontext = request.context
        assert self._app.has_session(sessioncontext.session_id)
        pipelines = self._app.get_pipelines(sessioncontext.session_id)
        return pb_core.PipelineListResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            pipeline_ids=list(pipelines),
        )

    def GetCreatePipelineResults(self, request, context):
        raise NotImplementedError  # TODO: GetCreatePipelineResults

    def GetExecutePipelineResults(self, request, context):
        raise NotImplementedError  # TODO: GetExecutePipelineResults


class DataflowService(pb_dataflow_grpc.DataflowExtServicer):
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
            execution_time=60.0,
        )


def main():
    logging.basicConfig(level=logging.INFO)

    D3mTa2('/tmp/vistrails_ta2').run()
