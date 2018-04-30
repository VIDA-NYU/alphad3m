import logging
import pandas as pd
from d3m_metadata.container.pandas import DataFrame
from d3metafeatureextraction import D3MetafeatureExtraction

logger = logging.getLogger(__name__)


def compute_metafeatures(dataset_path):
    df = DataFrame(pd.read_csv(dataset_path))
    names = df.columns.values
    df = df.rename(columns={"class": "target"})
    df.drop("d3mIndex", axis=1, inplace=True)
    metafeatures = D3MetafeatureExtraction(hyperparams=None).produce(inputs=df).value
    return metafeatures.values[0]
