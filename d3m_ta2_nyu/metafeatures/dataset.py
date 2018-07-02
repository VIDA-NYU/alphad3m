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
from pprint import pprint
logger = logging.getLogger(__name__)


class ComputeMetafeatures():

    def __init__(self, dataset, targets=None, features=None, DBSession=None):
        self.dataset = dataset
        self.db = DBSession()
        self.targets = targets
        self.features = features

    def _create_metafeatures_pipeline(self, origin):

        pipeline = database.Pipeline(
            origin=origin,
            dataset=self.dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            self.db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.4.18', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            self.db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        try:
            input_data = make_data_module('dataset')
            self.db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='targets', value=pickle.dumps(self.targets),
            ))
            self.db.add(database.PipelineParameter(
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

            self.db.add(pipeline)
            self.db.commit()
            print('PIPELINE ID: ', pipeline.id)
            return pipeline.id
        finally:
            self.db.close()

    def compute_metafeatures(self, origin):
        pipeline_id = self._create_metafeatures_pipeline(origin)
        # Run training
        logger.info("Computing Metafeatures")
        try:
            dataset = Dataset.load(self.dataset)
            train_run, outputs = execute_train(self.db, pipeline_id, dataset)
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

                print('OUTPUTS VALUE ')
                pprint(metafeatures_values)
            return list(metafeatures_values.values())
        except Exception:
            logger.exception("Error running Metafeatures")
            sys.exit(1)
        assert train_run is not None

        self.db.close()

    # def compute_metafeatures_OLD(self, dataset_path, table_file):
    #     f = open(dataset_path)
    #     dataset_info = json.load(f)
    #     target_col = 'Class'
    #     for res in dataset_info['dataResources']:
    #         for col in res['columns']:
    #             if 'suggestedTarget' in col['role']:
    #                 target_col = col['colName']
    #                 break
    #     print(target_col)
    #     df = DataFrame(pd.read_csv(table_file))
    #     df = df.rename(columns={target_col: "target"})
    #     df.drop("d3mIndex", axis=1, inplace=True)
    #     metafeatures = MetafeatureExtractor(hyperparams=None).produce(inputs=df).value
    #     return metafeatures.values[0]
