import csv
import json
import logging
import os
import sys

from d3m_ta2_nyu.common import read_dataset


logger = logging.getLogger(__name__)


def test(pipeline_id, dataset, results_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    # Load data
    try:
        data = read_dataset(dataset,
                            schema=os.path.join(persist_dir, 'dataSchema.json'))
    except Exception:
        logger.exception("Error loading dataset")
        raise
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
        "Executing pipeline from d3m_ta2_nyu.train",  # reason
        sinks,  # sinks
        None,  # extra_info
    ]])
    result, = results

    if result.errors:
        logger.error("Errors running pipeline:\n%s",
                     '\n'.join('%d: %s' % p
                               for p in result.errors.items()))
        sys.exit(1)

    output = get_module(vt_pipeline, 'test_targets').id
    output = result.objects[output]
    output = output.get_input('InternalPipe')

    with open(os.path.join(persist_dir, 'dataSchema.json')) as fp:
        column_name = [c['varName']
                       for c in json.load(fp)['trainData']['trainTargets']
                       if c['varName'] != 'd3mIndex'][0]

    with open(results_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['d3mIndex', column_name])

        for i, o in zip(data['testData']['index'], output):
            writer.writerow([i, o])
