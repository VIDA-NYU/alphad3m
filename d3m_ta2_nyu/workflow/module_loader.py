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
            # Create the classifier
            classifier = klass()
            # Train it
            classifier.fit(module_inputs['data'], module_inputs['targets'])
            cache.store('classifier', pickle.dumps(classifier))
            return {}
        elif global_inputs['run_type'] == 'test':
            classifier = pickle.loads(cache.get('classifier'))
            # Transform data
            predictions = classifier.predict(module_inputs['data'])
            return {'predictions': predictions}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _persisted_primitive(klass):
    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
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

            cache.store('hyperparams', pickle.dumps(hyperparams))
            cache.store('params', pickle.dumps(params))

            return {'data': results.value}
        elif global_inputs['run_type'] == 'test':
            # Load back the primitive
            hyperparams = pickle.loads(cache.get('hyperparams'))
            params = pickle.loads(cache.get('params'))
            primitive = klass(hyperparams=hyperparams_class(hyperparams))
            primitive.set_params(params=params)
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'])
            assert results.has_finished
            return {'data': results.value}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _simple_primitive(klass):
    def wrapper(*, module_inputs, **kwargs):
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
