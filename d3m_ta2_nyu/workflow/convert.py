"""Convert to/from the JSON representation.
"""

import importlib
import pickle


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _add_step(steps, modules, params, module_to_step, mod):
    if mod.id in module_to_step:
        return module_to_step[mod.id]

    # Special case: the "dataset" module
    if mod.package == 'data' and mod.name == 'dataset':
        module_to_step[mod.id] = 'inputs.0'
        return 'inputs.0'
    elif mod.package != 'd3m':
        raise ValueError("Got unknown module '%s:%s'", mod.package, mod.name)

    # Recursively walk upstream modules (to get `steps` in topological order)
    # Add inputs to a dictionary, in deterministic order
    inputs = {}
    for conn in sorted(mod.connections_to, key=lambda c: c.to_input_name):
        step = _add_step(steps, modules, params, module_to_step,
                         modules[conn.from_module_id])
        inputs[conn.to_input_name] = '%s.%s' % (step, conn.from_output_name)

    klass = get_class(mod.name)
    primitive_desc = {
        key: value
        for key, value in klass.metadata.query().items()
        if key in {'id', 'version', 'python_path', 'name', 'digest'}
    }

    # Create step description
    step = {
        'type': 'PRIMITIVE',
        'primitive': primitive_desc,
        'arguments': {
            name: {
                'type': 'CONTAINER',
                'data': data,
            }
            for name, data in inputs.items()
        },
        'outputs': [
            {'id': 'produce'},
        ],
    }

    # If hyperparameters are set, export them
    if mod.id in params and 'hyperparams' in params[mod.id]:
        hyperparams = pickle.loads(params[mod.id]['hyperparams'])
        hyperparams = {
            k: {'type': 'VALUE', 'data': v}
            for k, v in hyperparams.items()
        }
        step['hyperparams'] = hyperparams

    step_nb = 'steps.%d' % len(steps)
    steps.append(step)
    module_to_step[mod.id] = step_nb
    return step_nb


def to_d3m_json(pipeline):
    """Converts a Pipeline to the JSON schema from metalearning working group.
    """
    steps = []
    modules = {mod.id: mod for mod in pipeline.modules}
    params = {}
    for param in pipeline.parameters:
        params.setdefault(param.module_id, {})[param.name] = param.value
    module_to_step = {}
    for mod in modules.values():
        _add_step(steps, modules, params, module_to_step, mod)

    return {
        'id': str(pipeline.id),
        'schema': 'https://metadata.datadrivendiscovery.org/schemas/'
                  'v0/pipeline.json',
        'created': pipeline.created_date.isoformat() + 'Z',
        'context': 'TESTING',
        'inputs': [
            {'name': "input dataset"},
        ],
        'outputs': [
            {
                'data': 'steps.%d.produce' % (len(steps) - 1),
                'name': "predictions",
            }
        ],
        'steps': steps,
    }
