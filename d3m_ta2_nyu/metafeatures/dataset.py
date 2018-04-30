import logging
import pandas as pd
import json
import os
from d3m_metadata.container.pandas import DataFrame
from d3metafeatureextraction import D3MetafeatureExtraction

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
    df = DataFrame(pd.read_csv(table_file))
    names = df.columns.values
    df = df.rename(columns={target_col: "target"})
    df.drop("d3mIndex", axis=1, inplace=True)
    metafeatures = D3MetafeatureExtraction(hyperparams=None).produce(inputs=df).value
    return metafeatures.values[0]
