import logging
import os
import json
import pickle
import itertools

from d3m_ta2_nyu.workflow import database
from d3m import index
from d3m.container import Dataset, DataFrame, ndarray, matrix, List



# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
logger = logging.getLogger(__name__)

CONTAINER_CAST = {
    Dataset:{
        DataFrame:'d3m.primitives.data_transformation.dataset_to_dataframe.Common',
        ndarray: ('d3m.primitives.data_transformation.dataset_to_dataframe.Common'
                  '|d3m.primitives.data_transformation.dataframe_to_ndarray.Common'),
        matrix: "",
        List: ('d3m.primitives.data_transformation.dataset_to_dataframe.Common'
                  '|d3m.primitives.data_transformation.dataframe_to_list.Common')
    },
    DataFrame: {
        Dataset: "",
        ndarray: 'd3m.primitives.data_transformation.dataframe_to_ndarray.Common',
        matrix: "",
        List: 'd3m.primitives.data_transformation.dataframe_to_list.Common'
    },
    ndarray: {
        Dataset: "",
        DataFrame: 'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
        matrix: "",
        List: 'd3m.primitives.data_transformation.ndarray_to_list.Common'
    },
    matrix: {
        Dataset: "",
        DataFrame: "",
        ndarray: "",
        List: ""
    },
    List: {
        Dataset: "",
        DataFrame: 'd3m.primitives.data_transformation.list_to_dataframe.Common',
        ndarray: 'd3m.primitives.data_transformation.list_to_ndarray.Common',
        matrix: ""
    }
}


def make_pipeline_module(db, pipeline, name, package='d3m', version='2019.10.10'):
    pipeline_module = database.PipelineModule(pipeline=pipeline, package=package, version=version, name=name)
    db.add(pipeline_module)
    return pipeline_module


def make_data_module(db, pipeline, targets, features):
    input_data = make_pipeline_module(db, pipeline, 'dataset', 'data', '0.0')
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=input_data,
        name='targets', value=pickle.dumps(targets),
    ))
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=input_data,
        name='features', value=pickle.dumps(features),
    ))
    return input_data


def connect(db, pipeline, from_module, to_module, from_output='produce', to_input='inputs'):
    if not from_module.name.startswith('dataset'):
        from_module_primitive = index.get_primitive(from_module.name)
        from_module_output = from_module_primitive.metadata.query()['primitive_code']['class_type_arguments'][
            'Outputs']
    else:
        from_module_output = Dataset

    to_module_primitive = index.get_primitive(to_module.name)
    to_module_input = to_module_primitive.metadata.query()['primitive_code']['class_type_arguments'][
        'Inputs']

    if from_module_output != to_module_input:
        cast_module_steps = CONTAINER_CAST[from_module_output][to_module_input]
        if cast_module_steps:
            for cast_step in cast_module_steps.split('|'):
                cast_module = make_pipeline_module(db, pipeline,cast_step)
                db.add(database.PipelineConnection(pipeline=pipeline,
                                                   from_module=from_module,
                                                   to_module=cast_module,
                                                   from_output_name=from_output,
                                                   to_input_name=to_input))
                from_module = cast_module
        else:
            raise TypeError('Incompatible connection types: %s and %s'%(str(from_module_output),str(to_module_input)))

    db.add(database.PipelineConnection(pipeline=pipeline,
                                       from_module=from_module,
                                       to_module=to_module,
                                       from_output_name=from_output,
                                       to_input_name=to_input))


def set_hyperparams(db, pipeline, module, **hyperparams):
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=module,
        name='hyperparams', value=pickle.dumps(hyperparams),
    ))


def change_default_hyperparams(db, pipeline, primitive_name, primitive):
    if primitive_name == 'd3m.primitives.data_cleaning.imputer.SKlearn':
        set_hyperparams(db, pipeline, primitive, strategy='most_frequent')
    elif primitive_name == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
        set_hyperparams(db, pipeline, primitive, handle_unknown='ignore')
    elif primitive_name == 'd3m.primitives.learner.collaborative_filtering_link_prediction.DistilCollaborativeFiltering':
        set_hyperparams(db, pipeline, primitive, metric='meanAbsoluteError')
    elif primitive_name == 'd3m.primitives.data_preprocessing.text_reader.DataFrameCommon':
        set_hyperparams(db, pipeline, primitive, return_result='new')


def need_d3mindex(primitives):
    for primitive in primitives:
        if primitive in {'d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX',
                         'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX'}:
            return True
    return False


class BaseBuilder:

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        # TODO parameters 'features and 'targets' are not needed
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)
        try:

            #                        Denormalize
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /    |      \
            #                     /      |        \
            #                   /        |          \
            # Extract (attribute)  Extract (target)  |
            #         |                  |           |
            #  [preprocessors]           |           |
            #         |                  |           |
            #          \                /           /
            #            \             /          /
            #             [classifier]          /
            #                       |         /
            #                   ConstructPredictions
            # TODO: Use pipeline input for this
            input_data = make_data_module(db, pipeline, targets, features)

            prev_step = None
            if pipeline_template:
                prev_steps = {}
                count_template_steps = 0
                for pipeline_step in pipeline_template['steps']:
                    if pipeline_step['type'] == 'PRIMITIVE':
                        step = make_pipeline_module(db, pipeline, pipeline_step['primitive']['python_path'])
                        prev_steps['steps.%d.produce' % (count_template_steps)] = step
                        count_template_steps += 1
                        if 'hyperparams' in pipeline_step:
                            hyperparams = {}
                            for hyper, desc in pipeline_step['hyperparams'].items():
                                hyperparams[hyper] = desc['data']
                            set_hyperparams(db, pipeline, step, **hyperparams)
                    else:
                        # TODO In the future we should be able to handle subpipelines
                        break
                    if prev_step:
                        for argument, desc in pipeline_step['arguments'].items():
                            connect(db, pipeline, prev_steps[desc['data']], step, to_input=argument)
                    else:
                        connect(db, pipeline, input_data, step, from_output='dataset')
                    prev_step = step

            # Check if ALphaD3M is trying to augment
            search_result = None
            if 'RESULT.' in primitives[0]:
                result_index = int(primitives[0].split('.')[1])
                if result_index < len(search_results):
                    search_result = search_results[result_index]
                primitives = primitives[1:]

            # Check if there is result to augment
            if search_result:
                step_aug = make_pipeline_module(db, pipeline,
                                                'd3m.primitives.data_augmentation.datamart_augmentation.Common')
                if prev_step:
                    connect(db, pipeline, prev_step, step_aug)
                else:
                    connect(db, pipeline, input_data, step_aug, from_output='dataset')
                set_hyperparams(db, pipeline, step_aug, search_result=search_result, system_identifier='NYU')
                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
                connect(db, pipeline, step_aug, step0)
            else:
                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
                if prev_step:
                    connect(db, pipeline, prev_step, step0)
                else:
                    connect(db, pipeline, input_data, step0, from_output='dataset')

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'column_parser.DataFrameCommon')
            connect(db, pipeline, step1, step2)

            step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')

            semantic_type_list = ['https://metadata.datadrivendiscovery.org/types/Attribute']
            if need_d3mindex(primitives):  # Some primitives need the 'd3mIndex', so we can't filter out it
                semantic_type_list.append('https://metadata.datadrivendiscovery.org/types/PrimaryKey')

            set_hyperparams(db, pipeline, step3, semantic_types=semantic_type_list)
            connect(db, pipeline, step2, step3)

            step4 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step4,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                            )
            connect(db, pipeline, step2, step4)

            step = prev_step = step3
            preprocessors = primitives[:-1]
            estimator = primitives[-1]

            for preprocessor in preprocessors:
                step = make_pipeline_module(db, pipeline, preprocessor)
                change_default_hyperparams(db, pipeline, preprocessor, step)
                connect(db, pipeline, prev_step, step)
                prev_step = step

                if 'feature_selection' in step.name:  # FIXME: Use the primitive family
                    connect(db, pipeline, step4, step, to_input='outputs')

            step5 = make_pipeline_module(db, pipeline, estimator)
            change_default_hyperparams(db, pipeline, estimator, step5)
            connect(db, pipeline, step, step5)
            connect(db, pipeline, step4, step5, to_input='outputs')

            step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step5, step6)
            connect(db, pipeline, step2, step6, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except:
            return None
        finally:
                db.close()

    @staticmethod
    def make_template(imputer, estimator, dataset, pipeline_template, targets, features, feature_types, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin="template(imputer=%s, estimator=%s)" % (imputer, estimator),
                                     dataset=dataset)

        try:
            #                          data
            #                            |
            #                        Denormalize
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /     |     \
            #                     /       |       \
            #                   /         |         \
            # Extract (attribute)  Extract (target)  |
            #         |                  |        Extract (target, index)
            #     [imputer]              |           |
            #         |                  |           |
            #    One-hot encoder         |           |
            #         |                  |           |
            #          \                /           /
            #            \            /           /
            #             [estimator]          /
            #                       |         /
            #                   ConstructPredictions
            # TODO: Use pipeline input for this
            input_data = make_data_module(db, pipeline, targets, features)

            prev_step = None
            if pipeline_template:
                prev_steps = {}
                count_template_steps = 0
                for pipeline_step in pipeline_template['steps']:
                    if pipeline_step['type'] == 'PRIMITIVE':
                        step = make_pipeline_module(db, pipeline, pipeline_step['primitive']['python_path'])
                        prev_steps['steps.%d.produce' % (count_template_steps)] = step
                        count_template_steps += 1
                        if 'hyperparams' in pipeline_step:
                            hyperparams = {}
                            for hyper, desc in pipeline_step['hyperparams'].items():
                                hyperparams[hyper] = desc['data']
                            set_hyperparams(db, pipeline, step, **hyperparams)
                    else:
                        # TODO In the future we should be able to handle subpipelines
                        break
                    if prev_step:
                        for argument, desc in pipeline_step['arguments'].items():
                            connect(db, pipeline, prev_steps[desc['data']], step, to_input=argument)
                    else:
                        connect(db, pipeline, input_data, step, from_output='dataset')
                    prev_step = step

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
            if prev_step:
                connect(db, pipeline, prev_step, step0)
            else:
                connect(db, pipeline, input_data, step0, from_output='dataset')

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'column_parser.DataFrameCommon')
            connect(db, pipeline, step1, step2)

            step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step3,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Attribute']
                            )
            connect(db, pipeline, step2, step3)

            step4 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step4,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                            )
            connect(db, pipeline, step2, step4)

            step5 = make_pipeline_module(db, pipeline, imputer)
            set_hyperparams(db, pipeline, step5, strategy='most_frequent')
            connect(db, pipeline, step3, step5)
            prev_step = None
            both = 0

            if 'integer' in feature_types or 'real' in feature_types:
                step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'extract_columns_by_semantic_types.DataFrameCommon')
                set_hyperparams(db, pipeline, step6,
                                semantic_types=['http://schema.org/Integer', 'http://schema.org/Float'])
                connect(db, pipeline, step5, step6)
                prev_step = step6
                both += 1

            if 'categorical' in feature_types:
                step7 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'extract_columns_by_semantic_types.DataFrameCommon')
                set_hyperparams(db, pipeline, step7,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/CategoricalData'])
                connect(db, pipeline, step5, step7)

                step8 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn')
                set_hyperparams(db, pipeline, step8, handle_unknown='ignore')
                connect(db, pipeline, step7, step8)
                prev_step = step8
                both += 1

            if both == 2:
                step9 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'horizontal_concat.DataFrameConcat')
                connect(db, pipeline, step6, step9, to_input='left')
                connect(db, pipeline, step8, step9, to_input='right')
                prev_step = step9

            if both == 0:  # There is not categorical neither numeric features
                step_fallback = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                                   'one_hot_encoder.SKlearn')
                set_hyperparams(db, pipeline, step_fallback, handle_unknown='ignore')
                connect(db, pipeline, step5, step_fallback)
                prev_step = step_fallback

            step10 = make_pipeline_module(db, pipeline, estimator)
            connect(db, pipeline, prev_step, step10)
            connect(db, pipeline, step4, step10, to_input='outputs')

            step11 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                        'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step10, step11)
            connect(db, pipeline, step2, step11, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        except:
            return None
        finally:
            db.close()

    @staticmethod
    def make_template_augment(datamart_system, imputer, estimator, dataset, pipeline_template, targets,
                              features, feature_types, search_result, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(
            origin="template(datamart_system=%s, imputer=%s, estimator=%s)" % (datamart_system, imputer, estimator),
            dataset=dataset)

        try:
            #                          data
            #                            |
            #                        Denormalize
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /     |     \
            #                     /       |       \
            #                   /         |         \
            # Extract (attribute)  Extract (target)  |
            #         |                  |        Extract (target, index)
            #     [imputer]              |           |
            #         |                  |           |
            #    One-hot encoder         |           |
            #         |                  |           |
            #          \                /           /
            #            \            /           /
            #             [estimator]          /
            #                       |         /
            #                   ConstructPredictions
            # TODO: Use pipeline input for this
            input_data = make_data_module(db, pipeline, targets, features)

            prev_step = None
            if pipeline_template:
                prev_steps = {}
                count_template_steps = 0
                for pipeline_step in pipeline_template['steps']:
                    if pipeline_step['type'] == 'PRIMITIVE':
                        step = make_pipeline_module(db, pipeline, pipeline_step['primitive']['python_path'])
                        prev_steps['steps.%d.produce' % (count_template_steps)] = step
                        count_template_steps += 1
                        if 'hyperparams' in pipeline_step:
                            hyperparams = {}
                            for hyper, desc in pipeline_step['hyperparams'].items():
                                hyperparams[hyper] = desc['data']
                            set_hyperparams(db, pipeline, step, **hyperparams)
                    else:
                        # TODO In the future we should be able to handle subpipelines
                        break
                    if prev_step:
                        for argument, desc in pipeline_step['arguments'].items():
                            connect(db, pipeline, prev_steps[desc['data']], step, to_input=argument)
                    else:
                        connect(db, pipeline, input_data, step, from_output='dataset')
                    prev_step = step
            step_aug = make_pipeline_module(db, pipeline,
                                            'd3m.primitives.data_augmentation.datamart_augmentation.Common')
            if prev_step:
                connect(db, pipeline, prev_step, step_aug)
            else:
                connect(db, pipeline, input_data, step_aug, from_output='dataset')
            set_hyperparams(db, pipeline, step_aug, search_result=search_result, system_identifier=datamart_system)

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
            connect(db, pipeline, step_aug, step0)

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'column_parser.DataFrameCommon')
            connect(db, pipeline, step1, step2)

            step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step3,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Attribute']
                            )
            connect(db, pipeline, step2, step3)

            step4 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step4,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                            )
            connect(db, pipeline, step2, step4)

            step5 = make_pipeline_module(db, pipeline, imputer)
            set_hyperparams(db, pipeline, step5, strategy='most_frequent')
            connect(db, pipeline, step3, step5)
            prev_step = None
            both = 0

            if 'integer' in feature_types or 'real' in feature_types:
                step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'extract_columns_by_semantic_types.DataFrameCommon')
                set_hyperparams(db, pipeline, step6,
                                semantic_types=['http://schema.org/Integer', 'http://schema.org/Float'])
                connect(db, pipeline, step5, step6)
                prev_step = step6
                both += 1

            if 'categorical' in feature_types:
                step7 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'extract_columns_by_semantic_types.DataFrameCommon')
                set_hyperparams(db, pipeline, step7,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/CategoricalData'])
                connect(db, pipeline, step5, step7)

                step8 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn')
                set_hyperparams(db, pipeline, step8, handle_unknown='ignore')
                connect(db, pipeline, step7, step8)
                prev_step = step8
                both += 1

            if both == 2:
                step9 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'horizontal_concat.DataFrameConcat')
                connect(db, pipeline, step6, step9, to_input='left')
                connect(db, pipeline, step8, step9, to_input='right')
                prev_step = step9

            if both == 0:  # There is not categorical neither numeric features
                step_fallback = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                                   'one_hot_encoder.SKlearn')
                set_hyperparams(db, pipeline, step_fallback, handle_unknown='ignore')
                connect(db, pipeline, step5, step_fallback)
                prev_step = step_fallback

            step10 = make_pipeline_module(db, pipeline, estimator)
            connect(db, pipeline, prev_step, step10)
            connect(db, pipeline, step4, step10, to_input='outputs')

            step11 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                        'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step10, step11)
            connect(db, pipeline, step2, step11, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id

        except:
            return None
        finally:
            db.close()

    @staticmethod
    def make_meanbaseline(origin, dataset, DBSession):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, 'targets', 'features')
            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.classification.'
                                                       'gaussian_classification.MeanBaseline')
            connect(db, pipeline, input_data, step0, from_output='dataset')
            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except:
            return None
        finally:
            db.close()

    TEMPLATES_AUGMENTATION = {
        'CLASSIFICATION': list(itertools.product(
            # DATAMART
            ['NYU'],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',
                'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
                'd3m.primitives.classification.decision_tree.SKlearn',
                'd3m.primitives.classification.gaussian_naive_bayes.SKlearn',
                'd3m.primitives.classification.gradient_boosting.SKlearn',
                'd3m.primitives.classification.linear_svc.SKlearn',
                'd3m.primitives.classification.logistic_regression.SKlearn',
                'd3m.primitives.classification.multinomial_naive_bayes.SKlearn',
                'd3m.primitives.classification.passive_aggressive.SKlearn',
                'd3m.primitives.classification.sgd.SKlearn',
            ],
        )),
        'DEBUG_CLASSIFICATION': list(itertools.product(
            # DATAMART
            ['NYU'],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',

            ],
        )),
        'REGRESSION': list(itertools.product(
            # DATAMART
            [ 'NYU'],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
                'd3m.primitives.regression.decision_tree.SKlearn',
                'd3m.primitives.regression.gaussian_process.SKlearn',
                'd3m.primitives.regression.gradient_boosting.SKlearn',
                'd3m.primitives.regression.lasso.SKlearn',
                'd3m.primitives.regression.passive_aggressive.SKlearn',
            ],
        )),
        'DEBUG_REGRESSION': list(itertools.product(
            # DATAMART
            ['NYU'],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
            ],
        )),
    }

    TEMPLATES = {
        'CLASSIFICATION': list(itertools.product(
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',
                'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
                'd3m.primitives.classification.decision_tree.SKlearn',
                'd3m.primitives.classification.gaussian_naive_bayes.SKlearn',
                'd3m.primitives.classification.gradient_boosting.SKlearn',
                'd3m.primitives.classification.linear_svc.SKlearn',
                'd3m.primitives.classification.logistic_regression.SKlearn',
                'd3m.primitives.classification.multinomial_naive_bayes.SKlearn',
                'd3m.primitives.classification.passive_aggressive.SKlearn',
                'd3m.primitives.classification.sgd.SKlearn',
            ],
        )),
        'DEBUG_CLASSIFICATION': list(itertools.product(
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',

            ],
        )),
        'REGRESSION': list(itertools.product(
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
                'd3m.primitives.regression.decision_tree.SKlearn',
                'd3m.primitives.regression.gaussian_process.SKlearn',
                'd3m.primitives.regression.gradient_boosting.SKlearn',
                'd3m.primitives.regression.lasso.SKlearn',
                'd3m.primitives.regression.passive_aggressive.SKlearn',
            ],
        )),
        'DEBUG_REGRESSION': list(itertools.product(
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
            ],
        )),
    }


class TimeseriesClassificationBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, targets, features)

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
            connect(db, pipeline, input_data, step0, from_output='dataset')

            if len(primitives) == 1:
                step1 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, step0, step1)
                connect(db, pipeline, step0, step1, to_input='outputs')
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except Exception as e:
            logger.error(e)
            return None
        finally:
            db.close()


class TimeseriesForecastingBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        def get_time_unit(name_col):
            name = name_col.lower()

            if name.startswith('y'):
                return 'Y'
            elif name.startswith('mon'):
                return 'M'
            elif name.startswith('w'):
                return 'W'
            elif name.startswith('d'):
                return 'D'
            elif name.startswith('h'):
                return 'h'
            elif name.startswith('min'):
                return 'm'
            elif name.startswith('s'):
                return 's'
            elif name.startswith('mil'):
                return 'ms'
            elif name.startswith('mic'):
                return 'us'
            elif name.startswith('n'):
                return 'ns'
            elif name.startswith('m'):  # Default form m  is minutes
                return 'm'

            return name_col

        def extract_hyperparameters(dataset_path):
            with open(dataset_path) as fin:
                dataset_json = json.load(fin)
            hyperparameters = {}

            for resource in dataset_json['dataResources']:
                filters = []
                indexes = []
                for column in resource['columns']:
                    if 'suggestedGroupingKey' in column['role']:
                        filters.append(column['colIndex'])
                    if 'timeIndicator' in column['role'] and column['colType'] in ['integer', 'float']:
                        hyperparameters['datetime_index_unit'] = get_time_unit(column['colName'])
                    if 'timeIndicator' in column['role'] and column['colType'] == 'dateTime':
                        indexes.append(column['colIndex'])
                        # TODO: Extract datetime_indexes, now there is a bug in the datasetdoc.json with these fields

                if len(filters) > 0:
                    hyperparameters['filter_index_two'] = filters[0]
                if len(filters) > 1:
                    hyperparameters['filter_index_one'] = filters[1]

            return hyperparameters

        distil_hyperparameters = extract_hyperparameters(dataset[7:])
        try:
            if len(primitives) == 1:
                input_data = make_data_module(db, pipeline, targets, features)

                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
                connect(db, pipeline, input_data, step0, from_output='dataset')

                step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_'
                                                           'transformation.dataset_to_dataframe.Common')
                connect(db, pipeline, step0, step1)

                step2 = make_pipeline_module(db, pipeline, primitives[0])
                if len(distil_hyperparameters) > 0:
                    set_hyperparams(db, pipeline, step2, **distil_hyperparameters)
                connect(db, pipeline, step1, step2)
                connect(db, pipeline, step1, step2, to_input='outputs')
                db.add(pipeline)
                db.commit()
                logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
                return pipeline.id
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id
        except:
            return None
        finally:
            db.close()


class CommunityDetectionBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            if len(primitives) == 1:
                input_data = make_data_module(db, pipeline, targets, features)
                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.community_detection.'
                                                           'community_detection_parser.CommunityDetectionParser')
                connect(db, pipeline, input_data, step0, from_output='dataset')

                step1 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, step0, step1)

                db.add(pipeline)
                db.commit()
                logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
                return pipeline.id
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id
        except:
            return None
        finally:
            db.close()


class LinkPredictionBuilder(BaseBuilder):
    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            if len(primitives) == 1:  # Not working since we dont produce more than one item in outputs: produce_target
                input_data = make_data_module(db, pipeline, targets, features)

                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'load_single_graph.DistilSingleGraphLoader')
                connect(db, pipeline, input_data, step0, from_output='dataset')

                step1 = make_pipeline_module(db, pipeline, primitives[0])

                connect(db, pipeline, step0, step1)
                connect(db, pipeline, step0, step1, to_input='outputs', from_output='produce_target')

                db.add(pipeline)
                db.commit()
                logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
                return pipeline.id
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id
        except:
            return None
        finally:
            db.close()


class GraphMatchingBuilder(BaseBuilder):
    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            if len(primitives) == 1:  # Not working since we dont produce more than one item in outputs
                input_data = make_data_module(db, pipeline, targets, features)
                step0 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, input_data, step0, from_output='dataset')
                db.add(pipeline)
                db.commit()
                logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
                return pipeline.id
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id
        except:
            return None
        finally:
            db.close()


class VertexNominationBuilder(BaseBuilder):
    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            if len(primitives) == 1:  # Not working since we dont produce more than one item in outputs
                input_data = make_data_module(db, pipeline, targets, features)
                step0 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, input_data, step0, from_output='dataset')
                db.add(pipeline)
                db.commit()
                logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
                return pipeline.id
            else:
                pipeline_id = super().make_d3mpipeline(primitives, origin, dataset, search_results, pipeline_template,
                                                       targets, features, DBSession=DBSession)
                return pipeline_id
        except:
            return None
        finally:
            db.close()


class ObjectDetectionBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, targets, features)

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
            connect(db, pipeline, input_data, step0, from_output='dataset')

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step2,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                            'https://metadata.datadrivendiscovery.org/types/FileName']
                            )
            connect(db, pipeline, step1, step2)

            step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step3,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                            )
            connect(db, pipeline, step1, step3)

            step4 = make_pipeline_module(db, pipeline, primitives[0])
            connect(db, pipeline, step2, step4)
            connect(db, pipeline, step3, step4, to_input='outputs')

            step5 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step4, step5)
            connect(db, pipeline, step2, step5, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except:
            return None
        finally:
            db.close()


class AudioBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, targets, features)

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
            connect(db, pipeline, input_data, step0, from_output='dataset')

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'column_parser.DataFrameCommon')
            connect(db, pipeline, step1, step2)

            step3 = make_pipeline_module(db, pipeline, primitives[0])
            connect(db, pipeline, input_data, step3, from_output='dataset')

            step = prev_step = step3
            preprocessors = primitives[1:-1]
            estimator = primitives[-1]

            for preprocessor in preprocessors:
                step = make_pipeline_module(db, pipeline, preprocessor)
                change_default_hyperparams(db, pipeline, preprocessor, step)
                connect(db, pipeline, prev_step, step)
                prev_step = step

            step4 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step4,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                            )
            connect(db, pipeline, step2, step4)

            if 'feature_selection' in step.name:  # FIXME: Use the primitive family
                connect(db, pipeline, step4, step, to_input='outputs')

            step5 = make_pipeline_module(db, pipeline, estimator)
            change_default_hyperparams(db, pipeline, estimator, step5)
            connect(db, pipeline, step, step5)
            connect(db, pipeline, step4, step5, to_input='outputs')

            step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step5, step6)
            connect(db, pipeline, step2, step6, to_input='reference')

            db.add(pipeline)
            db.commit()

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except:
            return None
        finally:
            db.close()