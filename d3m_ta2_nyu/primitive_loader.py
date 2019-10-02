"""Class that retrieves installed D3M primitives.
"""
import os
import logging
import json
from d3m import index

logger = logging.getLogger(__name__)


PRIMITIVES_INFO_COM_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_info_com.json')
PRIMITIVES_INFO_SUM_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_info_sum.json')


black_list = {
    'd3m.primitives.data_preprocessing.audio_loader.DistilAudioDatasetLoader',
    'd3m.primitives.data_preprocessing.audio_reader.BBN',
    'd3m.primitives.data_preprocessing.audio_reader.DataFrameCommon',
    'd3m.primitives.data_preprocessing.audio_slicer.Umich',
    'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX'
    'd3m.primitives.data_preprocessing.do_nothing.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing.DSBOX',
    'd3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_preprocessing.image_reader.DataFrameCommon'
    'd3m.primitives.data_preprocessing.signal_dither.BBN',
    'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
    'd3m.primitives.data_preprocessing.truncated_svd.SKlearn',
    'd3m.primitives.data_preprocessing.vertical_concatenate.DSBOX',
    'd3m.primitives.data_preprocessing.video_reader.DataFrameCommon'
    'd3m.primitives.data_transformation.add_semantic_types.DataFrameCommon',
    'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
    'd3m.primitives.data_transformation.cast_to_type.Common',
    'd3m.primitives.data_transformation.collaborative_filtering_parser.CollaborativeFilteringParser',
    'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
    'd3m.primitives.data_transformation.conditioner.Conditioner',
    'd3m.primitives.data_transformation.conditioner.StaticEnsembler',
    'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon',
    'd3m.primitives.data_transformation.cut_audio.DataFrameCommon',
    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
    'd3m.primitives.data_transformation.denormalize.Common',
    'd3m.primitives.data_transformation.edge_list_to_graph.EdgeListToGraph',
    'd3m.primitives.data_transformation.extract_columns.DataFrameCommon',
    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
    'd3m.primitives.data_transformation.extract_columns_by_structural_types.DataFrameCommon',
    'd3m.primitives.data_transformation.fast_ica.SKlearn',
    'd3m.primitives.data_transformation.graph_matching_parser.GraphMatchingParser',
    'd3m.primitives.data_transformation.graph_node_splitter.GraphNodeSplitter',
    'd3m.primitives.data_transformation.graph_to_edge_list.DSBOX',
    'd3m.primitives.data_transformation.graph_to_edge_list.GraphToEdgeList',
    'd3m.primitives.data_transformation.graph_transformer.GraphTransformer',
    'd3m.primitives.data_transformation.laplacian_spectral_embedding.JHU',
    'd3m.primitives.data_transformation.list_to_dataframe.Common',
    'd3m.primitives.data_transformation.list_to_ndarray.Common',
    'd3m.primitives.data_transformation.load_graphs.DistilGraphLoader',
    'd3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
    'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
    'd3m.primitives.data_transformation.remove_semantic_types.DataFrameCommon',
    'd3m.primitives.data_transformation.replace_semantic_types.DataFrameCommon',
    'd3m.primitives.data_transformation.segment_curve_fitter.BBN',
    'd3m.primitives.data_transformation.sequence_to_bag_of_tokens.BBN',
    'd3m.primitives.data_transformation.simple_column_parser.DataFrameCommon',
    'd3m.primitives.data_transformation.stack_ndarray_column.Common',
    'd3m.primitives.data_transformation.stacking_operator.StackingOperator',
    'd3m.primitives.data_transformation.to_numeric.DSBOX',
    'd3m.primitives.data_transformation.vertex_classification_parser.VertexClassificationParser',
    'd3m.primitives.data_transformation.zero_count.ZeroCount'
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
    def get_primitives_info_summarized():
        """
        Returns a dictionary grouping primitive names by family and associating each primitive to a distinct number
        """

        if os.path.isfile(PRIMITIVES_INFO_SUM_PATH):
            with open(PRIMITIVES_INFO_SUM_PATH) as fin:
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

            with open(PRIMITIVES_INFO_SUM_PATH, 'w') as fout:
                json.dump(primitives, fout, indent=4)
            logger.info('Loading primitives info from D3M index')

        return primitives

    @staticmethod
    def get_primitives_info_complete():
        if os.path.isfile(PRIMITIVES_INFO_COM_PATH):
            with open(PRIMITIVES_INFO_COM_PATH) as fin:
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

            with open(PRIMITIVES_INFO_COM_PATH, 'w') as fout:
                json.dump(primitives, fout, indent=4)
            logger.info('Loading primitives info from D3M index')

        return primitives
