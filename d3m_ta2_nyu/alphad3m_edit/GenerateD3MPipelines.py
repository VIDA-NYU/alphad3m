import logging
import os
import pickle
from d3m_ta2_nyu.workflow import database
from d3m.container import Dataset

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

        try:
            #                          data
            #                            |
            #                       -Denormalize-
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /     |     \
            #                     /       |       \
            #                   /         |         \
            # Extract (attribute)  Extract (target)  |
            #         |               |              |
            #     CastToType      CastToType         |
            #         |               |              |
            #     [imputer]           |             /
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

            # FIXME: Denormalize?
            #step0 = make_primitive_module('.datasets.Denormalize')
            #connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('.datasets.DatasetToDataFrame')
            connect(input_data, step1, from_output='dataset')
            #connect(step0, step1)

            step2 = make_primitive_module('.data.ColumnParser')
            connect(step1, step2)

            step3 = make_primitive_module('.data.'
                                          'ExtractColumnsBySemanticTypes')
            set_hyperparams(
                step3,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            )
            connect(step2, step3)

            step4 = make_primitive_module('.data.CastToType')
            connect(step3, step4)

            step = prev_step = step4
            preprocessors = primitives[:-1]
            classifier = primitives[-1]
            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                connect(prev_step, step)
                prev_step = step

            step6 = make_primitive_module('.data.'
                                          'ExtractColumnsBySemanticTypes')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target',
                ],
            )
            connect(step2, step6)

            step7 = make_primitive_module('.data.CastToType')
            connect(step6, step7)

            step8 = make_primitive_module(classifier)
            connect(step, step8)
            connect(step7, step8, to_input='outputs')

            step9 = make_primitive_module('.data.ConstructPredictions')
            connect(step8, step9)
            connect(step2, step9, to_input='reference')

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
            primitives = ["d3m.primitives.bbn.time_series.AudioReader", "d3m.primitives.bbn.time_series.ChannelAverager",
                          "d3m.primitives.bbn.time_series.SignalDither", "d3m.primitives.bbn.time_series.SignalFramer",  "d3m.primitives.bbn.time_series.SignalMFCC",
                          "d3m.primitives.bbn.time_series.UniformSegmentation", "d3m.primitives.bbn.time_series.SegmentCurveFitter", "d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans",
                          "d3m.primitives.bbn.time_series.SignalFramer", "d3m.primitives.bbn.time_series.SequenceToBagOfTokens", "d3m.primitives.bbn.time_series.BBNTfidfTransformer",
                          "d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier"]
            step0 = make_primitive_module("d3m.primitives.bbn.time_series.TargetsReader")
            connect(input_data, step0, from_output='dataset')

            step = prev_step = step0
            preprocessors = []
            if len(primitives) > 1:
                preprocessors = primitives[0:len(primitives) - 1]
            classifier = primitives[len(primitives) - 1]
            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                connect(prev_step, step)
                prev_step = step

            # step1 = make_primitive_module('.datasets.DatasetToDataFrame')
            # connect(input_data, step1, from_output='dataset')
            #
            # step2 = make_primitive_module('.data.ExtractTargets')
            # connect(step1, step2)
            #
            # step3 = make_primitive_module('.data.CastToType')
            # connect(step2, step3)

            step4 = make_primitive_module(classifier)
            connect(step, step4)
            #connect(step3, step4, to_input='outputs')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_graphMatching_pipeline_from_strings(origin, dataset, targets=None, features=None, DBSession=None):
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
            step0 = make_primitive_module("d3m.primitives.jhu_primitives.SeededGraphMatching")
            connect(input_data, step0, from_output='dataset')
            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_communityDetection_pipeline_from_strings(origin, dataset, targets=None, features=None,
                                                      DBSession=None):
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
            step0 = make_primitive_module("d3m.primitives.sri.graph.CommunityDetectionParser")
            connect(input_data, step0, from_output='dataset')
            step1 = make_primitive_module('d3m.primitives.sri.psl.CommunityDetection')
            connect(step0, step1)

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        finally:
            db.close()

    @staticmethod
    def make_image_regression_pipeline_from_strings(origin, dataset, targets=None, features=None,
                                                    DBSession=None):
        logger.info('MAKING IMAGE REGRESSION PIPELINE')
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
            primitives = ["d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                          "d3m.primitives.dsbox.DataFrameToTensor", "d3m.primitives.dsbox.Vgg16ImageFeature",
                          "d3m.primitives.sklearn_wrap.SKPCA", "d3m.primitives.sklearn_wrap.SKRandomForestRegressor"]
            step0 = make_primitive_module("d3m.primitives.dsbox.Denormalize")
            connect(input_data, step0, from_output='dataset')

            step = prev_step = step0
            preprocessors = []
            if len(primitives) > 1:
                preprocessors = primitives[0:len(primitives) - 1]
            classifier = primitives[len(primitives) - 1]
            for preprocessor in preprocessors:
                step = make_primitive_module(preprocessor)
                connect(prev_step, step)
                prev_step = step

            # step1 = make_primitive_module('.datasets.DatasetToDataFrame')
            # connect(input_data, step1, from_output='dataset')
            #
            # step2 = make_primitive_module('.data.ExtractTargets')
            # connect(step1, step2)
            #
            # step3 = make_primitive_module('.data.CastToType')
            # connect(step2, step3)

            step4 = make_primitive_module(classifier)
            connect(step, step4)
            # connect(step3, step4, to_input='outputs')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        finally:
            db.close()

