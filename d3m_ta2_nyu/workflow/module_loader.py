import importlib
import pickle


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _data(*, global_inputs, **kwargs):
    return {'data': global_inputs['data']}


def _targets(*, global_inputs, **kwargs):
    return {'targets': global_inputs.get('targets')}


def _sklearn(name):
    klass = get_class(name)

    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
        if global_inputs['run_type'] == 'train':
            # Get classifier from module parameter
            if 'hyperparams' in module_inputs:
                classifier = pickle.loads(module_inputs['hyperparams'][0])
            # Use default hyperparameters
            else:
                classifier = klass()
            # Train it
            classifier.fit(module_inputs['data'][0],
                           module_inputs['targets'][0])
            cache.store('classifier', pickle.dumps(classifier))
            return {}
        elif global_inputs['run_type'] == 'test':
            classifier = pickle.loads(cache.get('classifier'))
            # Transform data
            predictions = classifier.predict(module_inputs['data'][0])
            return {'predictions': predictions}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _persisted_primitive(klass):
    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
        # Find the hyperparams class
        import typing
        hyperparams_class = typing.get_type_hints(klass.__init__)['hyperparams']

        # Get hyperparameters from module parameter
        if 'hyperparams' in module_inputs:
            hyperparams = pickle.loads(module_inputs['hyperparams'][0])
            hyperparams = hyperparams_class(hyperparams)
        # Use default hyperparams
        else:
            hyperparams = hyperparams_class.defaults()

        if global_inputs['run_type'] == 'train':
            # Create the primitive
            primitive = klass(hyperparams=hyperparams)
            # Train the primitive
            primitive.set_training_data(inputs=module_inputs['data'][0])
            primitive.fit()
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'][0])
            assert results.has_finished

            # Persist the parameters
            params = primitive.get_params()
            cache.store('params', pickle.dumps(params))

            return {'data': results.value}
        elif global_inputs['run_type'] == 'test':
            # Load back the primitive
            params = pickle.loads(cache.get('params'))
            primitive = klass(hyperparams=hyperparams)
            primitive.set_params(params=params)
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'][0])
            assert results.has_finished
            return {'data': results.value}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _simple_primitive(klass):
    def wrapper(*, module_inputs, **kwargs):
        # Find the hyperparams class
        import typing
        hyperparams_class = typing.get_type_hints(klass.__init__)['hyperparams']

        # Get hyperparameters from module parameter
        if 'hyperparams' in module_inputs:
            hyperparams = pickle.loads(module_inputs['hyperparams'][0])
            hyperparams = hyperparams_class(hyperparams)
        # Use default hyperparams
        else:
            hyperparams = hyperparams_class.defaults()

        # Create the primitive
        primitive = klass(hyperparams=hyperparams)
        # Transform data
        results = primitive.produce(inputs=module_inputs['data'][0])
        assert results.has_finished
        return {'data': results.value}

    return wrapper


def _loader_primitives(name):
    from primitive_interfaces.transformer import TransformerPrimitiveBase

    klass = get_class(name)
    if issubclass(klass, TransformerPrimitiveBase):
        return _simple_primitive(klass)
    else:
        return _persisted_primitive(klass)


_loaders = {'sklearn-builtin': _sklearn,
            'primitives': _loader_primitives,
            'data': {'data': _data, 'targets': _targets}.get}


def loader(package, version, name):
    return _loaders[package](name)
