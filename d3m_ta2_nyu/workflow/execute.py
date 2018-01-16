"""Pipeline execution system.
"""

import logging
from sqlalchemy import not_
from sqlalchemy.orm import joinedload

from . import database
from . import module_loader


logger = logging.getLogger(__name__)


class Cache(object):
    def __init__(self, db, run, module, from_run_id=None):
        self.db = db
        self.run = run
        self.module = module
        self.from_run_id = from_run_id

    def store(self, key, value):
        if not isinstance(value, bytes):
            raise TypeError("Cache expected bytes, not '%s'" % type(value))
        logger.info("Cache: %s recording %s", self.module.id, key)
        self.db.add(database.Output(run=self.run, module=self.module,
                                   output_name=key, value=value))

    def get(self, key):
        # TODO: record Input?
        if self.from_run_id is not None:
            return (
                self.db.query(database.Output)
                    .filter(database.Output.run_id == self.from_run_id)
                    .filter(database.Output.module_id == self.module.id)
                    .filter(database.Output.output_name == key)
            ).one().value
        else:
            return (
                self.db.query(database.Output)
                .filter(not_(database.Output.run.special))
                .filter(database.Output.run.pipeline_id ==
                        self.run.pipeline_id)
                .filter(database.Output.module_id == self.module.id)
                .filter(database.Output.output_name == key)
            ).one().value


def cache_from_run(from_run_id):
    def wrapper(db, run, module):
        return Cache(db, run, module, from_run_id)
    return wrapper


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


def execute_train(db, pipeline, data, targets, crossval=False):
    pipeline = get_pipeline(db, pipeline)

    global_inputs = {'data': data,
                     'targets': targets,
                     'run_type': 'train'}

    if crossval:
        return execute(db, pipeline, module_loader.loader,
                       "Training for cross-validation",
                       run_type=database.RunType.TRAIN,
                       module_inputs={},
                       global_inputs=global_inputs,
                       special=True)
    else:
        return execute(db, pipeline, module_loader.loader,
                       "Training",
                       run_type=database.RunType.TRAIN,
                       module_inputs={},
                       global_inputs=global_inputs)


def execute_test(db, pipeline, data, crossval=False,
                 from_training_run_id=None):
    pipeline = get_pipeline(db, pipeline)

    module_inputs = {}  # TODO: Get module inputs from database
    global_inputs = {'data': data,
                     'run_type': 'test'}

    if crossval:
        assert from_training_run_id is not None
        return execute(db, pipeline, module_loader.loader,
                       "Testing for cross-validation",
                       run_type=database.RunType.TEST,
                       module_inputs=module_inputs,
                       global_inputs=global_inputs,
                       cache=cache_from_run(from_training_run_id),
                       special=True)
    else:
        return execute(db, pipeline, module_loader.loader,
                       "Testing",
                       run_type=database.RunType.TEST,
                       module_inputs=module_inputs,
                       global_inputs=global_inputs)


def execute(db, pipeline, module_loader, reason,
            run_type, module_inputs={}, global_inputs={},
            cache=None,
            special=False):
    pipeline = get_pipeline(db, pipeline)

    logger.info("Executing pipeline %s", pipeline.id)

    # Load the modules
    modules = {}
    dependencies = {mod.id: set() for mod in pipeline.modules}
    dependents = {mod.id: set() for mod in pipeline.modules}
    for module in pipeline.modules:
        modules[module.id] = module, module_loader(module.package,
                                                   module.version,
                                                   module.name)

        for conn in module.connections_to:
            dependencies[module.id].add(conn.from_module_id)
            dependents[conn.from_module_id].add(module.id)

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

    if cache is None:
        cache = Cache

    # Execute
    outputs = []
    while to_execute:
        executed = []
        for mod_id in to_execute:
            if not all(dep in outputs for dep in dependencies[mod_id]):
                continue

            module, function = modules[mod_id]
            logger.info("Executing module %s (%s %s)",
                        mod_id, module.package, module.name)

            # Build inputs
            module_inputs = dict(module_inputs.get(mod_id, {}))
            for conn in module.connections_to:
                if conn.to_input_name not in module_inputs:
                    module_inputs[conn.to_input_name] = \
                        outputs[conn.from_module_id][conn.from_output_name]

            try:
                outputs[mod_id] = function(
                    module_inputs=module_inputs,
                    global_inputs=global_inputs,
                    cache=cache(db, pipeline, module),
                )
            except Exception:
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

    # Get output from sinks
    global_outputs = {}
    for mod_id, down in dependents.items():
        if not down:
            global_outputs[mod_id] = outputs[mod_id]
    if len(global_outputs) > 1:
        logger.warning("More than one sinks (%d)", len(global_outputs))

    return run.id, global_outputs
