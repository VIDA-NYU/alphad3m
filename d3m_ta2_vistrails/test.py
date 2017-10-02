import logging
import numpy
import sys
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator
from vistrails.core.modules.module_registry import get_module_registry
from vistrails.core.utils import DummyView
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails.common import read_dataset


logger = logging.getLogger(__name__)


# FIXME: Duplicate code
def get_module(pipeline, label):
    for module in pipeline.module_list:
        if '__desc__' in module.db_annotations_key_index:
            name = module.get_annotation_by_key('__desc__').value
            if name == label:
                return module
    return None


def test(vt_file, dataset, persist_dir, results_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    from userpackages.simple_persist import configuration as persist_config
    from userpackages.simple_persist.init import Internal

    # Load file
    # Copied from VistrailsApplicationInterface#open_vistrail()
    locator = BaseLocator.from_url(vt_file)
    loaded_objs = vistrails.core.db.io.load_vistrail(locator)
    controller = VistrailController(loaded_objs[0], locator,
                                    *loaded_objs[1:])
    controller.select_latest_version()
    vt_pipeline = controller.current_pipeline.do_copy()

    # Load data
    data = read_dataset(dataset)
    logger.info("Loaded dataset, columns: %r", data['testData']['columns'])

    test_data = data['testData']['frame']

    # Set input to Internal modules
    Internal.values = {
        get_module(vt_pipeline, 'test_data').id: test_data,
    }

    # Load persisted data
    persist_config.file_store = persist_dir

    # Find Persist modules, remove upstream connections
    registry = get_module_registry()
    descr = registry.get_descriptor_by_name('org.vistrails.vistrails.persist',
                                            'Persist')
    for module in vt_pipeline.module_list:
        if module.module_descriptor == descr:
            upstream = vt_pipeline.graph.inverse_adjacency_list[module.id]
            for _, conn_id in upstream:
                vt_pipeline.delete_connection(conn_id)

    # Select the sink
    sinks = [get_module(vt_pipeline, 'test_targets').id]

    results, changed = controller.execute_workflow_list([[
        controller.locator,  # locator
        controller.current_version,  # version
        vt_pipeline,  # pipeline
        DummyView(),  # view
        None,  # custom_aliases
        None,  # custom_params
        "Executing pipeline from d3m_ta2_vistrails.train",  # reason
        sinks,  # sinks
        None,  # extra_info
    ]])
    result, = results

    if result.errors:
        logger.error("Errors running pipeline:\n%s",
                     '\n'.join('%d: %s' % p
                               for p in result.errors.iteritems()))
        sys.exit(1)

    output = get_module(vt_pipeline, 'test_targets').id
    output = result.objects[output]
    output = output.get_input('InternalPipe')

    numpy.savetxt(results_path, output, delimiter=',', header="prediction")
