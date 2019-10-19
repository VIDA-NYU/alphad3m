import logging
import os
import json
import pickle
import itertools

from d3m_ta2_nyu.workflow import database


# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
logger = logging.getLogger(__name__)


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


class BaseBuilder:

    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
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
            set_hyperparams(db, pipeline, step3,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Attribute']
                            )

            connect(db, pipeline, step2, step3)

            step = prev_step = step3
            preprocessors = primitives[:-1]
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
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
                db.close()

    @staticmethod
    def make_text_pipeline_from_strings(primitives, origin, dataset, targets=None, features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:

            input_data = make_data_module('dataset')
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='targets', value=pickle.dumps(targets),
            ))
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='features', value=pickle.dumps(features),
            ))
            step0 = make_primitive_module('d3m.primitives.data_transformation.denormalize.Common')
            connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(step0, step1)

            step2 = make_primitive_module('d3m.primitives.data_transformation.column_parser.DataFrameCommon')
            connect(step1, step2)

            step3 = make_primitive_module('d3m.primitives.data_transformation'
                                          '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step3,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/Attribute']
            )
            connect(step2, step3)

            step4 = make_primitive_module('d3m.primitives.data_preprocessing.text_reader.DataFrameCommon')
            set_hyperparams(
                step4,
                return_result='new'
            )
            connect(step3, step4)

            step6 = make_primitive_module('d3m.primitives.data_transformation'
                                          '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target'
                ],
            )
            connect(step2, step6)

            step7 = make_primitive_module('d3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn')
            connect(step4, step7)

            step8 = make_primitive_module(primitives[-1])
            connect(step7, step8)
            connect(step6, step8, to_input='outputs')

            step9 = make_primitive_module('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
            connect(step8, step9)
            connect(step2, step9, to_input='reference')
            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        finally:
            db.close()

    @staticmethod
    def make_image_pipeline_from_strings(primitives, origin, dataset, targets=None, features=None, DBSession=None):
        db = DBSession()

        pipeline = database.Pipeline(
            origin=origin,
            dataset=dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:

            input_data = make_data_module('dataset')
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='targets', value=pickle.dumps(targets),
            ))
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='features', value=pickle.dumps(features),
            ))
            step0 = make_primitive_module('d3m.primitives.data_transformation.denormalize.Common')
            connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(step0, step1)

            step2 = make_primitive_module('d3m.primitives.data_preprocessing.image_reader.DataFrameCommon')
            connect(step1, step2)

            step3 = make_primitive_module(
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(step3, semantic_types=['http://schema.org/ImageObject'])
            connect(step2, step3)

            step4 = make_primitive_module(
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(step4, semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
            connect(step2, step4)

            step5 = make_primitive_module('d3m.primitives.data_transformation.dataframe_to_ndarray.Common')
            connect(step3, step5)

            step6 = make_primitive_module('d3m.primitives.feature_extraction.vgg16.Umich')
            connect(step5, step6)

            step7 = make_primitive_module('d3m.primitives.data_transformation.ndarray_to_dataframe.Common')
            connect(step6, step7)

            step8 = make_primitive_module(primitives[-1])
            connect(step7, step8)
            connect(step4, step8, to_input='outputs')

            step9 = make_primitive_module('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
            connect(step8, step9)
            connect(step1, step9, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        finally:
            db.close()

    @staticmethod
    def make_audio_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
        db = DBSession()

        pipeline = database.Pipeline(
            origin=origin,
            dataset=dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        try:

            input_data = make_data_module('dataset')
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='targets', value=pickle.dumps(targets),
            ))
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='features', value=pickle.dumps(features),
            ))

            def set_hyperparams(module, **hyperparams):
                db.add(database.PipelineParameter(
                    pipeline=pipeline, module=module,
                    name='hyperparams', value=pickle.dumps(hyperparams),
                ))

            primitives = ['d3m.primitives.data_preprocessing.channel_averager.BBN',
                          'd3m.primitives.data_preprocessing.signal_dither.BBN',
                          'd3m.primitives.time_series_segmentation.signal_framer.BBN',
                          'd3m.primitives.feature_extraction.signal_mfcc.BBN',
                          'd3m.primitives.time_series_segmentation.uniform_segmentation.BBN',
                          'd3m.primitives.data_transformation.segment_curve_fitter.BBN',
                          'd3m.primitives.clustering.cluster_curve_fitting_kmeans.BBN',
                          'd3m.primitives.time_series_segmentation.signal_framer.BBN',
                          'd3m.primitives.data_transformation.sequence_to_bag_of_tokens.BBN',
                          'd3m.primitives.feature_extraction.tfidf_vectorizer.BBN',
                          'd3m.primitives.classification.mlp.BBN']
            step0 = make_primitive_module('d3m.primitives.data_preprocessing.targets_reader.BBN')
            connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('d3m.primitives.data_preprocessing.audio_reader.BBN')
            connect(input_data, step1, from_output='dataset')

            step = prev_step = step1
            preprocessors = []
            if len(primitives) > 1:
                preprocessors = primitives[0:len(primitives) - 1]
            classifier = primitives[len(primitives) - 1]
            check_clustered = False
            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                if 'signal_mfcc' in preprocessor:
                    set_hyperparams(step, num_ceps=3)
                elif 'cluster_curve_fitting_kmeans' in preprocessor:
                    set_hyperparams(step, n_clusters=512)
                    check_clustered = True
                elif 'signal_framer' in preprocessor and check_clustered:
                    set_hyperparams(step, frame_length_s=1.0, frame_shift_s=1.0)
                elif 'tfidf_vectorizer' in preprocessor:
                    set_hyperparams(step, sublinear_tf=True)
                connect(prev_step, step)
                prev_step = step

            step2 = make_primitive_module(classifier)
            connect(step, step2)
            connect(step0, step2, to_input='outputs')

            step3 = make_primitive_module('d3m.primitives.data_transformation.denormalize.Common')
            connect(input_data, step3, from_output='dataset')

            step4 = make_primitive_module('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(step3, step4)

            step5 = make_primitive_module('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
            connect(step2, step5)
            connect(step4, step5, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_template(imputer, estimator, dataset, pipeline_template, targets, features, DBSession=None):
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

            step4 = make_pipeline_module(db, pipeline, imputer)
            set_hyperparams(db, pipeline, step4, strategy='most_frequent')
            connect(db, pipeline, step3, step4)

            step5 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn')
            set_hyperparams(db, pipeline, step5, handle_unknown='ignore')
            connect(db, pipeline, step4, step5)

            step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step6,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Target']
                            )
            connect(db, pipeline, step2, step6)

            step7 = make_pipeline_module(db, pipeline, estimator)
            connect(db, pipeline, step5, step7)
            connect(db, pipeline, step6, step7, to_input='outputs')

            step8 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step8,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Target',
                                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
                                            ]
                            )
            connect(db, pipeline, step2, step8)

            step9 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step7, step9)
            connect(db, pipeline, step8, step9, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_template_augment(datamart_system, imputer, estimator, dataset, pipeline_template, targets,
                              features, search_result, DBSession=None):
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
            set_hyperparams(db, pipeline, step3, semantic_types=[
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                ]
            )
            connect(db, pipeline, step2, step3)

            step4 = make_pipeline_module(db, pipeline, imputer)
            set_hyperparams(db, pipeline, step4, strategy='most_frequent')
            connect(db, pipeline, step3, step4)

            step5 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn')
            set_hyperparams(db, pipeline, step5, handle_unknown='ignore')
            connect(db, pipeline, step4, step5)

            step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step6, semantic_types=[
                'https://metadata.datadrivendiscovery.org/types/Target',
                ]
            )
            connect(db, pipeline, step2, step6)

            step7 = make_pipeline_module(db, pipeline, estimator)
            connect(db, pipeline, step5, step7)
            connect(db, pipeline, step6, step7, to_input='outputs')

            step8 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(db, pipeline, step8, semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey']
            )
            connect(db, pipeline, step2, step8)

            step9 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                       'construct_predictions.DataFrameCommon')
            connect(db, pipeline, step7, step9)
            connect(db, pipeline, step8, step9, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id
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
                step1 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
                connect(db, pipeline, step0, step1)

                step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'column_parser.DataFrameCommon')
                connect(db, pipeline, step1, step2)

                step = prev_step = step2
                preprocessors = primitives[:-1]
                classifier = primitives[-1]

                for preprocessor in preprocessors:
                    step = make_pipeline_module(db, pipeline, preprocessor)
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

                step5 = make_pipeline_module(db, pipeline, classifier)
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

                step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
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
        finally:
            db.close()


class LinkPredictionBuilder(BaseBuilder):
    def make_d3mpipeline(self, primitives, origin, dataset, search_results, pipeline_template, targets=None,
                         features=None, DBSession=None):
        db = DBSession()
        pipeline = database.Pipeline(origin=origin, dataset=dataset)

        try:
            if len(primitives) == 1:  # Not working since we dont produce more than one item in outputs
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
        finally:
            db.close()


class CollaborativeFilteringBuilder(BaseBuilder):
    pass


class SemisupervisedClassificationBuilder(BaseBuilder):
    pass


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
        finally:
            db.close()

