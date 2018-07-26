import importlib
import logging
import os
import pickle


logger = logging.getLogger(__name__)


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _dataset(*, global_inputs, module_inputs, **kwargs):
    from d3m.container import Dataset
    from d3m.metadata.base import ALL_ELEMENTS

    TYPE_TARGET = 'https://metadata.datadrivendiscovery.org/types/Target'
    TYPE_T_TARGET = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
    TYPE_ATTRIBUTE = 'https://metadata.datadrivendiscovery.org/types/Attribute'

    # Get dataset from global inputs
    dataset = global_inputs['dataset']

    # Get list of targets and attributes from global inputs
    [targets] = module_inputs['targets']
    targets = pickle.loads(targets)
    [attributes] = module_inputs['features']
    attributes = pickle.loads(attributes)

    # Update dataset metadata from those
    dataset = Dataset(dataset)
    not_found = set(targets)
    if attributes is not None:
        not_found.update(attributes)
    for resID, res in dataset.items():
        if not hasattr(res, 'columns'):
            continue
        for col_index, col_name in enumerate(res.columns):
            types = set(
                dataset.metadata.query([resID,
                                        ALL_ELEMENTS,
                                        col_index])['semantic_types'])
            if (resID, col_name) in targets:
                types.add(TYPE_TARGET)
                types.add(TYPE_T_TARGET)
                types.discard(TYPE_ATTRIBUTE)
                not_found.discard((resID, col_name))
            elif attributes is None:
                types.discard(TYPE_TARGET)
                types.discard(TYPE_T_TARGET)
            elif (resID, col_name) in attributes:
                types.add(TYPE_ATTRIBUTE)
                types.discard(TYPE_TARGET)
                types.discard(TYPE_T_TARGET)
                not_found.discard((resID, col_name))
            else:
                types.discard(TYPE_ATTRIBUTE)
                types.discard(TYPE_TARGET)
                types.discard(TYPE_T_TARGET)
            dataset.metadata = dataset.metadata.update(
                [resID, ALL_ELEMENTS, col_index],
                {'semantic_types': tuple(types)})

    if not_found:
        logger.warning("Specified targets/attributes missing from dataset: %r",
                       not_found)

    return {'dataset': dataset}


def _primitive_arguments(primitive, method):
    methods = primitive.metadata.query()['primitive_code']['instance_methods']
    return set(methods[method]['arguments'])


def _loader_d3m_primitive(name, version):
    import d3m

    # Check the requested version against the d3m package version
    if version != d3m.__version__:
        raise ValueError("Invalid version %r != %r" % (
            version, d3m.__version__))

    # Get the class from the module name
    # FIXME: Use primitive UUIDs?
    klass = get_class(name)

    # Map static data
    if 'D3M_PRIMITIVE_STATIC' in os.environ:
        statics = os.environ['D3M_PRIMITIVE_STATIC']
        if not os.path.exists(statics):
            logger.error("$D3M_PRIMITIVE_STATIC points to non-existent path")
        volumes = {
            installation['key']: os.path.join(statics,
                                              installation['file_digest'])
            for installation in klass.metadata.query().get('installation', [])
            if installation['type'] in ['FILE', 'TGZ']
        }
    else:
        volumes = {}

    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
        # Find the hyperparams class
        primitive_type_args = \
            klass.metadata.query()['primitive_code']['class_type_arguments']
        hyperparams_class = primitive_type_args['Hyperparams']

        # Get default hyperparams
        hyperparams = hyperparams_class.defaults()

        # Override hyperparamameters module parameter
        if 'hyperparams' in module_inputs:
            [custom_hyperparams] = module_inputs['hyperparams']
            custom_hyperparams = pickle.loads(custom_hyperparams)
            hyperparams = hyperparams_class(hyperparams,
                                            **custom_hyperparams)
        cstr_kwargs = dict(hyperparams=hyperparams)
        if volumes:
            cstr_kwargs['volumes'] = volumes

        if global_inputs['run_type'] == 'train':
            # Figure out which arguments are meant for which method
            training_args_names = _primitive_arguments(klass,
                                                       'set_training_data')
            training_args = {}
            produce_args_names = _primitive_arguments(klass,
                                                      'produce')
            produce_args = {}
            for param, [value] in module_inputs.items():
                if param == 'hyperparams':
                    continue
                used = False
                if param in training_args_names:
                    training_args[param] = value
                    used = True
                if param in produce_args_names:
                    produce_args[param] = value
                    used = True
                if not used:
                    logger.warning("Unused argument: %r", param)

            # Create the primitive
            primitive = klass(**cstr_kwargs)

            # Train the primitive
            primitive.set_training_data(**training_args)
            primitive.fit()

            # Transform data
            results = primitive.produce(**produce_args)
            assert results.has_finished

            # Persist the parameters
            params = primitive.get_params()
            cache.store('params', pickle.dumps(params))

            return {'produce': results.value}
        elif global_inputs['run_type'] == 'test':
            # Figure out which arguments are meant for produce
            produce_args_names = _primitive_arguments(klass,
                                                      'produce')
            produce_args = {}
            for param, [value] in module_inputs.items():
                if param == 'hyperparams':
                    continue
                used = False
                if param in produce_args_names:
                    produce_args[param] = value
                    used = True
                if not used:
                    logger.warning("Unused argument: %r", param)

            # Load back the primitive
            params = pickle.loads(cache.get('params'))
            primitive = klass(**cstr_kwargs)
            primitive.set_params(params=params)

            # Transform data
            results = primitive.produce(**produce_args)
            assert results.has_finished

            return {'produce': results.value}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


_data_modules = {'dataset': _dataset}

_loaders = {'d3m': _loader_d3m_primitive,
            'data': lambda name, **kw: _data_modules.get(name)}


def loader(package, version, name):
    return _loaders[package](name, version=version)
