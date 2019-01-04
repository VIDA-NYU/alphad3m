"""Class that retrieves installed D3M primitives.
"""
import logging
from d3m import index

logger = logging.getLogger(__name__)

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
    def get_primitives_dict():
        """
        Returns a dictionary grouping primitive names by family and associating each primitive to a distinct number
        """
        count = 1
        primitives_dictionary = {}
        for name in D3MPrimitives.INSTALLED_PRIMITIVES:
            family = D3MPrimitives.get_family(name)
            if family in primitives_dictionary:
                primitives_dictionary[family][name] = count
            else:
                primitives_dictionary[family] = {}
                primitives_dictionary[family][name] = count
            count += 1
        return primitives_dictionary







