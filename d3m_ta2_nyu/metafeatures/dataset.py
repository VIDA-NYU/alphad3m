import logging
import pandas as pd
import json
import os
import sys
import pickle
import frozendict
from d3m.container.pandas import DataFrame
from d3m.primitives.byudml.metafeature_extraction import MetafeatureExtractor
from d3m_ta2_nyu.workflow.execute import execute_train
from d3m_ta2_nyu.workflow import database
from d3m.container import Dataset
from d3m.metadata import base as metadata_base
logger = logging.getLogger(__name__)


class ComputeMetafeatures():

    def __init__(self, dataset, targets=None, features=None, DBSession=None):
        self.dataset = Dataset.load(dataset)
        self.dataset_uri = dataset
        self.DBSession = DBSession
        self.targets = targets
        self.features = features

    def _add_target_columns_metadata(self):
        for target in self.targets:
            resource_id = target[0]
            target_name = target[1]
            for column_index in range(self.dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']):
                if self.dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('name',
                                                                                                None) == target_name:
                    semantic_types = list(self.dataset.metadata.query(
                        (resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('semantic_types', []))

                    if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
                        semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                        self.dataset.metadata = self.dataset.metadata.update(
                            (resource_id, metadata_base.ALL_ELEMENTS, column_index),
                            {'semantic_types': semantic_types})

                    if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
                        semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                        self.dataset.metadata = self.dataset.metadata.update(
                            (resource_id, metadata_base.ALL_ELEMENTS, column_index),
                            {'semantic_types': semantic_types})

    def _create_metafeatures_pipeline(self, db, origin):

        pipeline = database.Pipeline(
            origin=origin,
            dataset=self.dataset_uri)

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
            return make_module('d3m', '2018.4.18', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        input_data = make_data_module('dataset')
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='targets', value=pickle.dumps(self.targets),
        ))
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='features', value=pickle.dumps(self.features),
        ))

        step0 = make_primitive_module('.datasets.Denormalize')
        connect(input_data, step0, from_output='dataset')

        step1 = make_primitive_module('.datasets.DatasetToDataFrame')
        connect(step0, step1)

        step2 = make_primitive_module('.data.ColumnParser')
        connect(step1, step2)

        step3 = make_primitive_module('d3m.primitives.byudml.metafeature_extraction.MetafeatureExtractor')
        connect(step2, step3)

        db.add(pipeline)
        db.flush()
        logger.info(origin + ' PIPELINE ID: %s', pipeline.id)
        return pipeline.id

    def compute_metafeatures(self, origin):
        db = self.DBSession()
        # Add true and suggested targets
        self._add_target_columns_metadata()
        # Create the metafeatures computing pipeline
        pipeline_id = self._create_metafeatures_pipeline(db, origin)
        # Run training
        logger.info("Computing Metafeatures")
        try:

            train_run, outputs = execute_train(db, pipeline_id, self.dataset)
            metafeatures_values = {}
            for key, value in outputs.items():
                metafeatures_results = value['produce'].metadata.query(())['data_metafeatures']
                for metafeatures_key, metafeatures_value in metafeatures_results.items():
                    if isinstance(metafeatures_value, frozendict.FrozenOrderedDict):
                        for k, v in metafeatures_value.items():
                            if 'primitive' not in k:
                                metafeatures_values[metafeatures_key+'_'+ k] = v
                    else:
                        metafeatures_values[metafeatures_key] = metafeatures_value
            return list(metafeatures_values.values())
        except Exception:
            logger.exception("Error running Metafeatures")
            sys.exit(1)
        finally:
            db.rollback()
            db.close()

