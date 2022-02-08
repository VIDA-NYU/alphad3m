# Using BYU's metafeatures extraction
# Follow installation instructions at the byu github
# https://github.com/byu-dml/metalearn

from metalearn import Metafeatures
import pandas as pd
from pprint import pprint
import math
import logging
import traceback
import sys

logger = logging.getLogger(__name__)

class ComputeMetafeatures:
    default_mf = {'Dimensionality': 0.015843429636533086,
                       'MaxCardinalityOfCategoricalFeatures': 1072.0,
                       'MaxCardinalityOfNumericFeatures': 1011.0,
                       'MaxCategoricalAttributeEntropy': 6.976921762797408,
                       'MaxKurtosisOfNumericFeatures': 8.231902220545834,
                       'MaxMeansOfNumericFeatures': 4576.038210624418,
                       'MaxNumericAttributeEntropy': 1.863685998841421,
                       'MaxSkewnessOfNumericFeatures': 2.502922807458578,
                       'MaxStdDevOfNumericFeatures': 2120.1152578482533,
                       'MeanCardinalityOfCategoricalFeatures': 539.5,
                       'MeanCardinalityOfNumericFeatures': 461.73333333333335,
                       'MeanCategoricalAttributeEntropy': 4.327386548299678,
                       'MeanKurtosisOfNumericFeatures': 2.1635671992401475,
                       'MeanMeansOfNumericFeatures': 643.608624905147,
                       'MeanNumericAttributeEntropy': 1.6486568255343295,
                       'MeanSkewnessOfNumericFeatures': 0.8461338023813748,
                       'MeanStdDevOfNumericFeatures': 328.49003930073235,
                       'MinCardinalityOfCategoricalFeatures': 7.0,
                       'MinCardinalityOfNumericFeatures': 17.0,
                       'MinCategoricalAttributeEntropy': 1.677851333801948,
                       'MinKurtosisOfNumericFeatures': 0.3308760915549813,
                       'MinMeansOfNumericFeatures': 0.2691276794035415,
                       'MinNumericAttributeEntropy': 1.1530866872284191,
                       'MinSkewnessOfNumericFeatures': -1.857520871847883,
                       'MinStdDevOfNumericFeatures': 0.025336178497455068,
                       'NumberOfCategoricalFeatures': 2,
                       'NumberOfFeatures': 17,
                       'NumberOfFeaturesWithMissingValues': 1,
                       'NumberOfInstances': 1073,
                       'NumberOfInstancesWithMissingValues': 18,
                       'NumberOfMissingValues': 18,
                       'NumberOfNumericFeatures': 15,
                       'PredEigen1': 5498930.9672156805,
                       'PredEigen2': 95723.12297846466,
                       'PredEigen3': 45419.62282695437,
                       'PredPCA1': 0.9656433360609853,
                       'PredPCA2': 0.016809521043669992,
                       'PredPCA3': 0.007975942300555823,
                       'Quartile1CategoricalAttributeEntropy': 3.0026189410508133,
                       'Quartile1KurtosisOfNumericFeatures': 0.5764423827165652,
                       'Quartile1MeansOfNumericFeatures': 7.265839235787512,
                       'Quartile1NumericAttributeEntropy': 1.550129827273282,
                       'Quartile1SkewnessOfNumericFeatures': 0.5594009134313469,
                       'Quartile1StdDevOfNumericFeatures': 1.6206097663393182,
                       'Quartile2CategoricalAttributeEntropy': 4.327386548299678,
                       'Quartile2KurtosisOfNumericFeatures': 1.2043926549869868,
                       'Quartile2MeansOfNumericFeatures': 205.42218080149115,
                       'Quartile2NumericAttributeEntropy': 1.6987883922053246,
                       'Quartile2SkewnessOfNumericFeatures': 0.9604917260139706,
                       'Quartile2StdDevOfNumericFeatures': 118.14051269101712,
                       'Quartile3CategoricalAttributeEntropy': 5.652154155548542,
                       'Quartile3KurtosisOfNumericFeatures': 3.288335419979105,
                       'Quartile3MeansOfNumericFeatures': 607.4389561975769,
                       'Quartile3NumericAttributeEntropy': 1.7834166981325819,
                       'Quartile3SkewnessOfNumericFeatures': 1.378286679363259,
                       'Quartile3StdDevOfNumericFeatures': 370.4705894463986,
                       'RatioOfCategoricalFeatures': 0.11764705882352941,
                       'RatioOfFeaturesWithMissingValues': 0.058823529411764705,
                       'RatioOfInstancesWithMissingValues': 0.016775396085740912,
                       'RatioOfMissingValues': 0.0009867880050435831,
                       'RatioOfNumericFeatures': 0.8823529411764706,
                       'StdevCardinalityOfCategoricalFeatures': 753.0687219636732,
                       'StdevCardinalityOfNumericFeatures': 321.857214718991,
                       'StdevKurtosisOfNumericFeatures': 2.170988197697653,
                       'StdevMeansOfNumericFeatures': 1174.6612047703775,
                       'StdevSkewnessOfNumericFeatures': 0.9666852623372723,
                       'StdevStdDevOfNumericFeatures': 539.5955739242856}
    
    def compute_metafeatures(self, X, y):
        try:
            self.metafeatures = Metafeatures().compute(X=X, Y=y, metafeature_ids=list(self.default_mf.keys()))
        except:
            logger.info('ERROR COMPUTING METAFEATURES - USING DEFAULT')
            traceback.print_exc(file=sys.stdout)
            return self.default_mf
        
        self.single_value_mf = {}
        for feature in self.default_mf.keys():
            if self.metafeatures.get(feature) is None:
                self.single_value_mf[feature] = 0
            else:
                v = self.metafeatures[feature]['value']
                if (v != v) or (v in ['NUMERIC_TARGETS']):
                    self.single_value_mf[feature] = 0
                else:
                    self.single_value_mf[feature] = v
                    
        logger.info("METAFEATURES %s %s", self.single_value_mf, len(self.single_value_mf))
        return self.single_value_mf

def main():
    from sklearn.datasets import load_iris
    data = load_iris()
    cm = ComputeMetafeatures()
    pprint(cm.compute_metafeatures(data.data, data.target))


if __name__== "__main__":
  main()
