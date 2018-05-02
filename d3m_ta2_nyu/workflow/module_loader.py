import importlib
import pandas
import pickle


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _dataset(*, global_inputs, module_inputs, **kwargs):
    from d3m.container import Dataset
    from d3m.metadata.base import ALL_ELEMENTS

    TYPE_TARGET = 'https://metadata.datadrivendiscovery.org/types/Target'
    TYPE_ATTRIBUTE = 'https://metadata.datadrivendiscovery.org/types/Attribute'

    # Get dataset from global inputs
    dataset = global_inputs['dataset']

    # Get list of targets and attributes from global inputs
    targets = global_inputs['targets']

    # Update dataset metadata from those
    dataset = Dataset(dataset)
    for resID, res in dataset:
        for col_index, col_name in enumerate(res.columns):
            if (resID, col_name) in targets:
                types = set(
                    res.metadata.query([resID,
                                        ALL_ELEMENTS,
                                        col_index])['semantic_types'])
                types.remove(TYPE_ATTRIBUTE)
                types.add(TYPE_TARGET)
                dataset.metadata = dataset.metadata.update(
                    [resID, ALL_ELEMENTS, col_index],
                    {'semantic_types': tuple(types)})

    return {'dataset': dataset}


def _get_columns(*, module_inputs, **kwargs):
    columns = pickle.loads(module_inputs['columns'][0])
    data = module_inputs['data'][0]
    # We have to -1 because column 0 is the index to pandas
    return {'data': data[columns]}


def _merge_columns(*, module_inputs, **kwargs):
    data = module_inputs['data']
    return {'data': pandas.concat(data, axis=1)}


def _sklearn_transformer(klass):
    # TODO: Need a way to know if module needs one column at a time or not
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    one_column_at_a_time = LabelBinarizer, LabelEncoder

    def wrapper(*, module_inputs, global_inputs, cache, **kwargs):
        if global_inputs['run_type'] == 'train':
            data = module_inputs['data'][0]
            if issubclass(klass, one_column_at_a_time):
                # Train and transform, one column at a time
                transformers = []
                for _ in range(len(data.columns)):
                    # Get transformer from module parameter
                    if 'hyperparams' in module_inputs:
                        transformer = pickle.loads(
                            module_inputs['hyperparams'][0])
                    # Use default hyperparameters
                    else:
                        transformer = klass()
                    transformers.append(transformer)
                results = pandas.concat((
                    pandas.DataFrame(transformers[i].fit_transform(data[col]),
                                     index=data.index)
                    for i, col in enumerate(data.columns)),
                    axis=1)
                cache.store('transformer', pickle.dumps(transformers))
            else:
                # Get transformer from module parameter
                if 'hyperparams' in module_inputs:
                    transformer = pickle.loads(
                        module_inputs['hyperparams'][0])
                # Use default hyperparameters
                else:
                    transformer = klass()
                # Train and transform
                results = transformer.fit_transform(data)
                results = pandas.DataFrame(results, index=data.index)
                cache.store('transformer', pickle.dumps(transformer))
            return {'data': results}
        elif global_inputs['run_type'] == 'test':
            data = module_inputs['data'][0]
            if issubclass(klass, one_column_at_a_time):
                # Transform data, one column at a time
                transformers = pickle.loads(cache.get('transformer'))
                results = pandas.concat((
                    pandas.DataFrame(transformers[i].transform(data[col]),
                                     index=data.index)
                    for i, col in enumerate(data.columns)),
                    axis=1)
            else:
                # Transform data
                transformer = pickle.loads(cache.get('transformer'))
                results = transformer.transform(data)
                results = pandas.DataFrame(results, index=data.index)
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
            data = module_inputs['data'][0]
            predictions = classifier.predict(data)
            predictions = pandas.DataFrame(predictions, index=data.index)
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
            'data': {'dataset': _dataset,
                     'get_columns': _get_columns,
                     'merge_columns': _merge_columns}.get}


def loader(package, version, name):
    return _loaders[package](name)
