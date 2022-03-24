from d3m_interface import AutoML as BaseAutoML

#  TODO: When we simplify the AutoML in d3m-interface to DockerAutoML and SingularityAutoML subclasses, we can inherit
#   directly from them


class DockerAutoML(BaseAutoML):

    def __init__(self, output_folder, resource_folder=None, grpc_port=None, verbose=False):
        """Create/instantiate an AutoMLContainer object

        :param output_folder: Path to the output directory
        :param resource_folder: Path to the directory where the resources are stored. This is needed only for some
            primitives that use pre-trained models, databases, etc.
        :param grpc_port: Port to be used by GRPC
        :param verbose: Whether or not to show all the logs from AutoML systems
        """

        automl_id = 'AlphaD3M'
        container_runtime = 'docker'
        BaseAutoML.__init__(self, output_folder, automl_id, container_runtime, resource_folder, grpc_port, verbose)


class SingularityAutoML(BaseAutoML):

    def __init__(self, output_folder, resource_folder=None, grpc_port=None, verbose=False):
        """Create/instantiate an AutoMLContainer object

        :param output_folder: Path to the output directory
        :param resource_folder: Path to the directory where the resources are stored. This is needed only for some
            primitives that use pre-trained models, databases, etc.
        :param grpc_port: Port to be used by GRPC
        :param verbose: Whether or not to show all the logs from AutoML systems
        """

        automl_id = 'AlphaD3M'
        container_runtime = 'singularity'
        BaseAutoML.__init__(self, output_folder, automl_id, container_runtime, resource_folder, grpc_port, verbose)
