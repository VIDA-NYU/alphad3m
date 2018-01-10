"""Pipeline execution system.
"""

import logging
from sqlalchemy.orm import joinedload

from . import database


logger = logging.getLogger(__name__)


def execute_workflow(DBSession, pipeline_id, module_loader, reason,
                     run_type, module_inputs={},
                     special=False):
    logger.info("Executing pipeline %s", pipeline_id)
    db = DBSession()
    try:
        # Load the pipeline
        pipeline = (
            db.query(database.Pipeline)
                .filter(database.Pipeline.id == pipeline_id)
                .options(joinedload(database.Pipeline.modules),
                         joinedload(database.Pipeline.connections))
        ).one()

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
        run = database.Run(pipeline_id=pipeline_id,
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
    finally:
        db.close()
