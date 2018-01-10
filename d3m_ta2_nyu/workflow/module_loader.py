import importlib
import pickle


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _data(*, global_inputs):
    return global_inputs['data']


def _targets(*, global_inputs):
    return global_inputs['targets']


def _sklearn_wrapper(klass):
    def wrapper(*, module_inputs, global_inputs):
        if global_inputs['run_type'] == 'train':
            # Create the classifier
            classifier = klass()
            # Train it
            classifier.fit(module_inputs['data'], module_inputs['targets'])
            return {'classifier': pickle.dumps(classifier)}
        elif global_inputs['run_type'] == 'test':
            classifier = pickle.loads(module_inputs['classifier'])
            # Transform data
            predictions = classifier.predict(module_inputs['data'])
            return {'predictions': predictions}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _persisted_primitive_wrapper(klass):
    def wrapper(*, module_inputs, global_inputs):
        # Find the hyperparams class
        import typing
        hyperparams_class = typing.get_type_hints(klass.__init__)['hyperparams']

        if global_inputs['run_type'] == 'train':
            # Create the primitive with default hyperparams
            hyperparams = hyperparams_class.defaults()
            primitive = klass(hyperparams=hyperparams)
            # Train the primitive
            primitive.set_training_data(inputs=module_inputs['data'])
            primitive.fit()
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'])
            assert results.has_finished

            # Break down the primitive into pickleable structures
            hyperparams = dict(primitive.hyperparams)
            params = primitive.get_params()

            return {'data': results.value,
                    'primitive_hyperparams': pickle.dumps(hyperparams),
                    'primitive_params': pickle.dumps(params)}
        elif global_inputs['run_type'] == 'test':
            # Load back the primitive
            hyperparams = pickle.loads(module_inputs['primitive_hyperparams'])
            params = pickle.loads(module_inputs['primitive_params'])
            primitive = klass(hyperparams=hyperparams_class(hyperparams))
            primitive.set_params(params)
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'])
            assert results.has_finished
            return {'data': results.value}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _simple_primitive_wrapper(klass):
    def wrapper(*, module_inputs):
        # Find the hyperparams class
        import typing
        hp_class = typing.get_type_hints(klass.__init__)['hyperparams']
        # Create the primitive with default hyperparams
        hp = hp_class.defaults()
        primitive = klass(hyperparams=hp)
        # Transform data
        results = primitive.produce(inputs=module_inputs['data'])
        assert results.has_finished
        return {'data': results.value}

    return wrapper


def _loader_sklearn(module):
    klass = get_class(module.name)
    return _sklearn_wrapper(klass)


def _loader_primitives(module):
    from primitive_interfaces.transformer import TransformerPrimitiveBase

    klass = get_class(module.name)
    if issubclass(klass, TransformerPrimitiveBase):
        return _simple_primitive_wrapper(klass)
    else:
        return _persisted_primitive_wrapper(klass)


_loaders = {'sklearn-builtin': _loader_sklearn,
            'primitives': _loader_primitives,
            'data': {'data': _data, 'targets': _targets}.get}


def loader(module):
    _loaders[module.package](module)
