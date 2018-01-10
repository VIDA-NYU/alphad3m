import importlib


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _data(*, global_inputs):
    return global_inputs['data']


def _targets(*, global_inputs):
    return global_inputs['targets']


def _sklearn_wrapper(klass):
    def wrapper(*, module_inputs, global_inputs, interpreter):
        if global_inputs['run_type'] == 'train':
            # Create the classifier
            classifier = klass()
            # Train it
            classifier.fit(module_inputs['data'], module_inputs['targets'])
            # TODO: save trained object
        elif global_inputs['run_type'] == 'test':
            # TODO: load trained object
            classifier = None
            # Transform data
            predictions = classifier.predict(module_inputs['data'])
            return {'predictions': predictions}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _primitive_wrapper(klass):
    def wrapper(*, module_inputs, global_inputs, interpreter):
        if global_inputs['run_type'] == 'train':
            # Find the hyperparams class
            import typing
            hp_class = typing.get_type_hints(klass.__init__)['hyperparams']
            # Create the primitive with default hyperparams
            hp = hp_class.defaults()
            primitive = klass(hyperparams=hp)
            # Train the primitive
            primitive.set_training_data(inputs=module_inputs['data'])
            primitive.fit()
            # TODO: save trained primitive
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'])
            assert results.has_finished
            return {'data': results.value}
        elif global_inputs['run_type'] == 'test':
            # TODO: load trained primitive
            primitive = None
            # Transform data
            results = primitive.produce(inputs=module_inputs['data'])
            assert results.has_finished
            return {'data': results.value}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _loader_sklearn(module):
    klass = get_class(module.name)
    return _sklearn_wrapper(klass)


def _loader_primitives(module):
    klass = get_class(module.name)
    return _primitive_wrapper(klass)


_loaders = {'sklearn-builtin': _loader_sklearn,
            'primitives': _loader_primitives,
            'data': {'data': _data, 'targets': _targets}.get}


def loader(module):
    _loaders[module.package](module)
