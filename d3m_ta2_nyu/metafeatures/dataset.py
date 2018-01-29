import os
import pandas as pd
from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures

def load_dataframe(df):
    df.fillna(method='ffill', inplace=True)
    X = df.values[:,0:-1]
    Y = df.filter([df.keys()[-1]]).astype('str').values.flatten()
    attributes = []
    for i in range(0,len(X[0])):
        attributes.append((df.keys()[i],str(type(X[0][i]))))
    attributes.append(('class', list(set(Y))))
    return X, Y, attributes

def extract_metafeatures(X,Y,attributes):
    metafeatures = {}
    features, time = SimpleMetafeatures().timed_compute(X,Y,attributes)
    print("simple metafeatures compute time: {}".format(time))
    total_time = time
    for key, value in features.items():
        metafeatures[key] = value

    features, time = StatisticalMetafeatures().timed_compute(X,Y,attributes)
    print("statistical metafeatures compute time: {}".format(time))
    total_time = total_time + time
    for key, value in features.items():
        metafeatures[key] = value

    features, time = InformationTheoreticMetafeatures().timed_compute(X,Y,attributes)
    print("information theoretic metafeatures compute time: {}".format(time))
    total_time = total_time + time
    for key, value in features.items():
        metafeatures[key] = value

    return metafeatures

def compute_metafeatures(dataset_path):
    df = pd.read_csv(dataset_path)
    X, Y, attributes = load_dataframe(df)
    metadata = extract_metafeatures(X, Y, attributes)
    return metadata
