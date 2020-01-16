"""Class that retrieves installed D3M primitives.
"""
import os
import logging
import json
from d3m import index

logger = logging.getLogger(__name__)


PRIMITIVES_BY_NAME_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_name.json')
PRIMITIVES_BY_TYPE_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_type.json')


black_list = {
    #d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI,
    'd3m.primitives.classification.lupi_rfsel.LupiRFSelClassifier',
    'd3m.primitives.classification.lupi_rfsel.LupiRFSelClassifier'
    'd3m.primitives.classification.lupi_rf.LupiRFClassifier',
    'd3m.primitives.classification.lupi_rfsel.LupiRFSelClassifier',
    'd3m.primitives.classification.lupi_svm.LupiSvmClassifier',
    'd3m.primitives.data_preprocessing.audio_loader.DistilAudioDatasetLoader',
    'd3m.primitives.data_preprocessing.audio_reader.BBN',
    'd3m.primitives.data_preprocessing.audio_reader.Common',
    'd3m.primitives.data_preprocessing.audio_slicer.Umich',
    'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX'
    'd3m.primitives.data_preprocessing.do_nothing.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_preprocessing.image_reader.Common'
    'd3m.primitives.data_preprocessing.signal_dither.BBN',
    'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
    'd3m.primitives.data_preprocessing.truncated_svd.SKlearn',
    'd3m.primitives.data_preprocessing.vertical_concatenate.DSBOX',
    'd3m.primitives.data_preprocessing.video_reader.Common',
    'd3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn',
    'd3m.primitives.data_preprocessing.text_reader.Common',
    'd3m.primitives.data_preprocessing.image_reader.Common',
    'd3m.primitives.data_preprocessing.horizontal_concat.DSBOX',
    'd3m.primitives.data_preprocessing.dataset_text_reader.DatasetTextReader',
    'd3m.primitives.data_preprocessing.dataset_sample.Common',
    'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX',
    'd3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter',
    'd3m.primitives.data_preprocessing.audio_reader.Common',
    'd3m.primitives.data_preprocessing.csv_reader.Common',
    'd3m.primitives.data_preprocessing.flatten.DataFrameCommon'
}


class D3MPrimitiveLoader():

    INSTALLED_PRIMITIVES = index.search()

    @staticmethod
    def get_primitive_class(name):
        """
        Returns the class object given a primitive name
        """
        return index.get_primitive(name)

    @staticmethod
    def get_family(name):
        """
        Returns the family (DATA_PREPROCESSING, CLASSIFICATION, REGRESSION ...) object given a primitive name
        """
        return D3MPrimitiveLoader.get_primitive_class(name).metadata.to_json_structure()['primitive_family']

    @staticmethod
    def get_primitive_names():
        """
        Returns a list with the name of the available primitives
        """
        return list(D3MPrimitiveLoader.INSTALLED_PRIMITIVES.keys())

    @staticmethod
    def get_primitives_by_type():
        """
        Returns a dictionary grouping primitive names by family and associating each primitive to a distinct number
        """

        if os.path.isfile(PRIMITIVES_BY_TYPE_PATH):
            with open(PRIMITIVES_BY_TYPE_PATH) as fin:
                primitives = json.load(fin)
            logger.info('Loading primitives info from file')
        else:
            primitives = {}
            count = 1
            for name in D3MPrimitiveLoader.INSTALLED_PRIMITIVES:
                if name in black_list:
                    continue
                try:
                    family = D3MPrimitiveLoader.get_family(name)
                except:
                    logger.error('No information about primitive %s', name)
                    family = 'N/A'
                if family in primitives:
                    primitives[family][name] = count
                else:
                    primitives[family] = {}
                    primitives[family][name] = count
                count += 1

            with open(PRIMITIVES_BY_TYPE_PATH, 'w') as fout:
                json.dump(primitives, fout, indent=4)
            logger.info('Loading primitives info from D3M index')

        return primitives

    @staticmethod
    def get_primitives_by_name():
        if os.path.isfile(PRIMITIVES_BY_NAME_PATH):
            with open(PRIMITIVES_BY_NAME_PATH) as fin:
                primitives = json.load(fin)
            logger.info('Loading primitives info from file')
        else:
            primitives = []

            for primitive_name in D3MPrimitiveLoader.INSTALLED_PRIMITIVES:
                if primitive_name in black_list:
                    continue
                try:
                    primitive_obj = index.get_primitive(primitive_name)
                except:
                    logger.error('Error loading primitive %s' % primitive_name)
                    continue

                if hasattr(primitive_obj, 'metadata'):
                    try:
                        primitive = {
                            'id': primitive_obj.metadata.query()['id'],
                            'name': primitive_obj.metadata.query()['name'],
                            'version': primitive_obj.metadata.query()['version'],
                            'python_path': primitive_obj.metadata.query()['python_path'],
                            'digest': primitive_obj.metadata.query()['digest']
                        }
                    except:
                        continue
                primitives.append(primitive)

            with open(PRIMITIVES_BY_NAME_PATH, 'w') as fout:
                json.dump(primitives, fout, indent=4)
            logger.info('Loading primitives info from D3M index')

        return primitives
