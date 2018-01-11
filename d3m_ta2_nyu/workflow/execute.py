"""Pipeline execution system.
"""

import logging
from sqlalchemy.orm import joinedload

from . import database
from . import module_loader


logger = logging.getLogger(__name__)


def get_pipeline(db, pipeline):
    if isinstance(pipeline, str):
        # Load the pipeline
        pipeline = (
            db.query(database.Pipeline)
                .filter(database.Pipeline.id == pipeline)
                .options(joinedload(database.Pipeline.modules),
                         joinedload(database.Pipeline.connections))
        ).one()
    return pipeline


def execute_training(db, pipeline, data, targets, crossval=False):
    pipeline = get_pipeline(db, pipeline)

    global_inputs = {'data': data,
                     'targets': targets,
                     'run_type': 'train'}

    if crossval:
        return execute(db, pipeline, module_loader.loader,
                       "Training for cross-validation",
                       run_type=database.RunType.TRAIN,
                       module_inputs={},
                       globals_inputs=global_inputs,
                       special=True)
    else:
        return execute(db, pipeline, module_loader.loader,
                       "Training",
                       run_type=database.RunType.TRAIN,
                       module_inputs={},
                       globals_inputs=global_inputs)


def execute_test(db, pipeline, data, crossval=False):
    pipeline = get_pipeline(db, pipeline)

    module_inputs = {}  # TODO: Get module inputs from database
    global_inputs = {'data': data,
                     'run_type': 'test'}

    if crossval:
        return execute(db, pipeline, module_loader.loader,
                       "Testing for cross-validation",
                       run_type=database.RunType.TEST,
                       module_inputs=module_inputs,
                       global_inputs=global_inputs,
                       special=True)
    else:
        return execute(db, pipeline, module_loader.loader,
                       "Testing",
                       run_type=database.RunType.TEST,
                       module_inputs=module_inputs,
                       global_inputs=global_inputs,
                       special=True)


def execute(db, pipeline, module_loader, reason,
            run_type, module_inputs={}, global_inputs={},
            special=False):
    pipeline = get_pipeline(db, pipeline)

    logger.info("Executing pipeline %s", pipeline.id)

    # Load the modules
    modules = {}
    dependencies = {}
    dependents = {}
    for module in pipeline.modules:
        modules[module.id] = module.module, module_loader(module.module)

        for conn in module.connections_to:
            dependencies.setdefault(module.id, set()) \
                .add(conn.from_module_id)
            dependents.setdefault(conn.from_module_id, set()) \
                .add(module.id)

    logger.info("Loaded %d modules", len(modules))

    to_execute = set(modules)

    # Create a run
    run = database.Run(pipeline_id=pipeline.id,
                       reason=reason,
                       special=special)
    logger.info("Created run %s", run.id)

    global_inputs = {
        'run_type': run_type,
    }

    # Execute
    outputs = []
    while to_execute:
        executed = []
        for mod_id in to_execute:
            if all(dep in outputs for dep in dependencies[mod_id]):
                module, function = modules[mod_id]
                logger.info("Executing module %s (%s %s)",
                            mod_id, module.package, module.name)
                try:
                    module_inputs = dict(module_inputs.get(mod_id, {}))
                    # TODO: get output from connected modules
                    outputs[mod_id] = function(
                        module_inputs=module_inputs,
                        global_inputs=global_inputs,
                    )
                except Exception as e:
                    logger.exception("Got exception running module %s",
                                     mod_id)
                    raise
            executed.append(mod_id)
        if not executed:
            logger.error("Couldn't execute any module, %d remain",
                         len(to_execute))
            raise RuntimeError("Couldn't execute every module")
        for mod_id in executed:
            to_execute.remove(mod_id)

    # TODO: record input and output

    return run.id
