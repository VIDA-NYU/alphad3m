"""Class that retrieves installed D3M primitives.
"""
import os
import logging
import json
from d3m import index

logger = logging.getLogger(__name__)


PRIMITIVES_INFO_COM_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_info_com.json')
PRIMITIVES_INFO_SUM_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_info_sum.json')


class D3MPrimitives():

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
        return D3MPrimitives.get_primitive_class(name).metadata.to_json_structure()['primitive_family']

    @staticmethod
    def get_primitive_names():
        """
        Returns a list with the name of the available primitives
        """
        return list(D3MPrimitives.INSTALLED_PRIMITIVES.keys())

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
            for name in D3MPrimitives.INSTALLED_PRIMITIVES:
                try:
                    family = D3MPrimitives.get_family(name)
                except:
                    logger.error('No information about primitive %s', name)
                    family = 'None'
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

            for primitive_name in D3MPrimitives.INSTALLED_PRIMITIVES:
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
