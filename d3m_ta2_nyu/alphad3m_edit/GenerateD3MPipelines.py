import logging
import os
import pickle
from d3m_ta2_nyu.workflow import database


# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
logger = logging.getLogger(__name__)


class GenerateD3MPipelines():
    @staticmethod
    def make_pipeline_from_strings(primitives, origin, dataset, targets=None, features=None, DBSession=None):

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

        def change_default_hyperparams(primitive_name, primitive):
            if primitive_name == 'd3m.primitives.data_cleaning.imputer.SKlearn':
                set_hyperparams(primitive, strategy='most_frequent')
            elif primitive_name == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
                set_hyperparams(primitive, handle_unknown='ignore')


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
            #         |               |              |
            #    <preprocess>     CastToType         |
            #         |               |              |
            #     CastToType          |             /
            #            \            /           /
            #             [classifier]          /
            #                       |         /
            #                   ConstructPredictions
            # TODO: Use pipeline input for this
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
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            )
            connect(step2, step3)

            step = prev_step = step3
            preprocessors = primitives[:-1]
            classifier = primitives[-1]

            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                change_default_hyperparams(preprocessor, step)
                connect(prev_step, step)
                prev_step = step

            step5 = make_primitive_module('d3m.primitives.data_transformation.cast_to_type.Common')
            connect(step, step5)
            set_hyperparams(
                step5,
                type_to_cast='float',
            )

            step6 = make_primitive_module('d3m.primitives.data_transformation'
                                          '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                ],
            )
            connect(step2, step6)

            step7 = make_primitive_module('d3m.primitives.data_transformation.cast_to_type.Common')
            connect(step6, step7)

            step8 = make_primitive_module(classifier)
            connect(step5, step8)
            connect(step7, step8, to_input='outputs')

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
    def make_text_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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

            step8 = make_primitive_module('d3m.primitives.classification.random_forest.SKlearn')
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
    def make_image_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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
            set_hyperparams(
                step2,
                return_result='new'
            )
            connect(step1, step2)

            step3 = make_primitive_module('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(step3, semantic_types=['http://schema.org/ImageObject'])
            connect(step2, step3)

            step4 = make_primitive_module('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(step4, semantic_types=['https://metadata.datadrivendiscovery.org/types/SuggestedTarget'])
            connect(step2, step4)

            step5 = make_primitive_module('d3m.primitives.data_transformation.dataframe_to_ndarray.Common')
            connect(step3, step5)

            step6 = make_primitive_module('d3m.primitives.feature_extraction.vgg16.Umich')
            connect(step5, step6)

            step7 = make_primitive_module('d3m.primitives.data_transformation.ndarray_to_dataframe.Common')
            connect(step6, step7)

            step8 = make_primitive_module('d3m.primitives.regression.random_forest.SKlearn')
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

            primitives = ['d3m.primitives.bbn.time_series.ChannelAverager',
                          'd3m.primitives.bbn.time_series.SignalDither',
                          'd3m.primitives.bbn.time_series.SignalFramer',
                          'd3m.primitives.bbn.time_series.SignalMFCC',
                          'd3m.primitives.bbn.time_series.UniformSegmentation',
                          'd3m.primitives.bbn.time_series.SegmentCurveFitter',
                          'd3m.primitives.bbn.time_series.ClusterCurveFittingKMeans',
                          'd3m.primitives.bbn.time_series.SignalFramer',
                          'd3m.primitives.bbn.time_series.SequenceToBagOfTokens',
                          'd3m.primitives.bbn.time_series.BBNTfidfTransformer',
                          'd3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier']
            step0 = make_primitive_module('d3m.primitives.bbn.time_series.TargetsReader')
            connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('d3m.primitives.bbn.time_series.AudioReader')
            connect(input_data, step1, from_output='dataset')

            step = prev_step = step1
            preprocessors = []
            if len(primitives) > 1:
                preprocessors = primitives[0:len(primitives) - 1]
            classifier = primitives[len(primitives) - 1]
            check_clustered = False
            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                if 'SignalMFCC' in preprocessor:
                    set_hyperparams(step, num_ceps=3)
                elif 'ClusterCurveFittingKMeans' in preprocessor:
                    set_hyperparams(step, n_clusters=512)
                    check_clustered = True
                elif 'SignalFramer' in preprocessor and check_clustered:
                    set_hyperparams(step, frame_length_s=1.0, frame_shift_s=1.0)
                elif 'BBNTfidfTransformer' in preprocessor:
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
    def make_objectdetection_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):

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

            step2 = make_primitive_module('d3m.primitives.data_transformation.column_parser.DataFrameCommon')
            set_hyperparams(
                step2,
                parse_semantic_types=["http://schema.org/Boolean",
                                "http://schema.org/Integer",
                                "http://schema.org/Float",
                                "https://metadata.datadrivendiscovery.org/types/FloatVector",
                                "http://schema.org/DateTime"
                                ]
            )
            connect(step1, step2)




            step6 = make_primitive_module('d3m.primitives.data_transformation'
                                          '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                ],
            )
            connect(step2, step6)

            step8 = make_primitive_module('d3m.primitives.object_detection.retina_net.JPLPrimitives')
            connect(step2, step8)
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
    def make_graphmatching_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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

            step2 = make_primitive_module('d3m.primitives.data_transformation.column_parser.DataFrameCommon')
            connect(step1, step2)

            step3 = make_primitive_module('d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction')
            set_hyperparams(step3, link_prediction_hyperparams="gANjc3JpLnBzbC5saW5rX3ByZWRpY3Rpb24KTGlua1ByZWRpY3Rpb25IeXBlcnBhcmFtcwpxACmBcQF9cQIoWAsAAABwc2xfb3B0aW9uc3EDWAAAAABxBFgQAAAAcG9zdGdyZXNfZGJfbmFtZXEFWAcAAABwc2xfZDNtcQZYDwAAAGFkbW1faXRlcmF0aW9uc3EHTegDWAsAAABtYXhfdGhyZWFkc3EISwBYCgAAAGp2bV9tZW1vcnlxCUc/6AAAAAAAAFgPAAAAdHJ1dGhfdGhyZXNob2xkcQpHPnrX8pq8r0hYEQAAAHByZWRpY3Rpb25fY29sdW1ucQtYBAAAAGxpbmtxDHViLg==")
            connect(input_data, step3, from_output='dataset')

            step4 = make_primitive_module('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
            set_hyperparams(step4, use_columns=[0, 1])

            connect(step3, step4)
            connect(step2, step4, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_communitydetection_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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
            step0 = make_primitive_module('d3m.primitives.community_detection.community_detection_parser.CommunityDetectionParser')
            connect(input_data, step0, from_output='dataset')
            step1 = make_primitive_module('d3m.primitives.classification.community_detection.CommunityDetection')
            set_hyperparams(
                step1,
                jvm_memory=0.8
            )
            connect(step0, step1)

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_linkprediction_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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
            step0 = make_primitive_module('d3m.primitives.data_transformation.graph_matching_parser.GraphMatchingParser')
            connect(input_data, step0, from_output='dataset')
            step1 = make_primitive_module('d3m.primitives.data_transformation.graph_transformer.GraphTransformer')
            connect(step0, step1)
            step2 = make_primitive_module('d3m.primitives.link_prediction.link_prediction.LinkPrediction')
            connect(step1, step2)
            step3 = make_primitive_module('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
            connect(step2, step3)
            connect(step2, step3, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_vertexnomination_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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

            step0 = make_primitive_module('d3m.primitives.data_transformation.vertex_nomination_parser.VertexNominationParser')
            connect(input_data, step0, from_output='dataset')
            step1 = make_primitive_module('d3m.primitives.classification.vertex_nomination.VertexNomination')
            connect(step0, step1)
            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_timeseries_class_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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

            step1 = make_primitive_module('d3m.primitives.time_series_classification.k_neighbors.Kanine')
            connect(step0, step1)
            connect(step0, step1, to_input='outputs')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        finally:
            db.close()

    @staticmethod
    def make_timeseries_fore_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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

            step2 = make_primitive_module('d3m.primitives.time_series_forecasting.arima.Parrot')
            connect(step1, step2)
            connect(step1, step2, to_input='outputs')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        finally:
            db.close()
