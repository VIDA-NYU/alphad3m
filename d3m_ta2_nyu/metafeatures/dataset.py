import logging
import pandas as pd
import json
import os
from d3m.container.pandas import DataFrame
from d3m.primitives.byudml.metafeature_extraction import MetafeatureExtractor

logger = logging.getLogger(__name__)


def compute_metafeatures(dataset_path, table_file):
    f = open(dataset_path)
    dataset_info = json.load(f)
    target_col = 'Class'
    for res in dataset_info['dataResources']:
        for col in res['columns']:
            if 'suggestedTarget' in col['role']:
                target_col = col['colName']
                break
    print(target_col)
    df = DataFrame(pd.read_csv(table_file))
    df = df.rename(columns={target_col: "target"})
    df.drop("d3mIndex", axis=1, inplace=True)
    metafeatures = MetafeatureExtractor(hyperparams=None).produce(inputs=df).value
    return metafeatures.values[0]
