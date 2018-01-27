import copy
import importlib
import pandas
import pickle


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _data(*, global_inputs, **kwargs):
    return {'data': global_inputs['data']}


def _targets(*, global_inputs, **kwargs):
    return {'targets': global_inputs.get('targets')}


def _get_columns(*, module_inputs, **kwargs):
    columns = pickle.loads(module_inputs['columns'][0])
    data = module_inputs['data'][0]
    # We have to -1 because column 0 is the index to pandas
    return {'data': data.iloc[:, [e - 1 for e in columns]]}


def _merge_columns(*, module_inputs, **kwargs):
    data = module_inputs['data']
    return {'data': pandas.concat(data, axis=1)}


def _sklearn_transformer(klass):
    # TODO: Need a way to know if module needs one column at a time or not
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    one_column_at_a_time = LabelBinarizer, LabelEncoder

    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
        if global_inputs['run_type'] == 'train':
            # Get transformer from module parameter
            if 'hyperparams' in module_inputs:
                transformer = pickle.loads(module_inputs['hyperparams'][0])
            # Use default hyperparameters
            else:
                transformer = klass()
            data = module_inputs['data'][0]
            if issubclass(klass, one_column_at_a_time):
                # Train and transform, one column at a time
                transformers = [copy.copy(transformer)
                                for _ in range(len(data.columns))]
                results = pandas.concat((
                    pandas.DataFrame(transformers[i].fit_transform(data[col]))
                    for i, col in enumerate(data.columns)),
                    axis=1)
                cache.store('transformer', pickle.dumps(transformers))
            else:
                # Train and transform
                results = transformer.fit_transform(data)
                cache.store('transformer', pickle.dumps(transformer))
            return {'data': results}
        elif global_inputs['run_type'] == 'test':
            data = module_inputs['data'][0]
            if issubclass(klass, one_column_at_a_time):
                # Transform data, one column at a time
                transformers = pickle.loads(cache.get('transformer'))
                results = pandas.concat((
                    pandas.DataFrame(transformers[i].transform(data[col]))
                    for i, col in enumerate(data.columns)),
                    axis=1)
            else:
                # Transform data
                transformer = pickle.loads(cache.get('transformer'))
                results = transformer.transform(data)
            return {'data': results}
        else:
            raise ValueError("Global 'run_type' not set")

    return wrapper


def _sklearn_classifier(klass):
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


def _loader_sklearn(name):
    from sklearn.base import TransformerMixin

    klass = get_class(name)
    if issubclass(klass, TransformerMixin):
        return _sklearn_transformer(klass)
    else:
        return _sklearn_classifier(klass)


def _primitive_persisted(klass):
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


def _primitive_simple(klass):
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
        return _primitive_simple(klass)
    else:
        return _primitive_persisted(klass)


_loaders = {'sklearn-builtin': _loader_sklearn,
            'primitives': _loader_primitives,
            'data': {'data': _data, 'targets': _targets,
                     'get_columns': _get_columns,
                     'merge_columns': _merge_columns}.get}


def loader(package, version, name):
    return _loaders[package](name)
