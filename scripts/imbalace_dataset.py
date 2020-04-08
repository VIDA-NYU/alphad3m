import pandas as pd
import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


class NEW_ENN:
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features

    def fit_resample(self, X, y):
        n_features = X.shape[1]
        continuous_features = np.setdiff1d(np.arange(n_features), self.categorical_features)
        X_continuous = X[:, continuous_features]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_categorical = X[:, categorical_features]

        if X_continuous.dtype.name != 'object':
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64

        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore', dtype=dtype_ohe)
        X_ohe = ohe.fit_transform(X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical)
        X_encoded = sparse.hstack((X_continuous, X_ohe), format='csr')

        enn_balancer = EditedNearestNeighbours(sampling_strategy='all')
        X_resampled, y_resampled = enn_balancer.fit_resample(X_encoded, y)
        selected_indices = enn_balancer.sample_indices_
        X_resampled = X[selected_indices, :]

        return X_resampled, y_resampled


target_name = 'CASE_STATUS'
categorical_features = [0, 1,2,3,4,13,15,16,17,21,22,23,24,25,26,27,28]
#target_name = 'Hall_of_Fame'
#categorical_features = [0, 16]
index_name = 'd3mIndex'

data = pd.read_csv('/Users/rlopez/D3M/others/learningData_visa.csv')
X = data.drop([index_name, target_name], axis=1)
y = data[target_name]
print(sorted(Counter(y).items()))


over_sampling = SMOTENC(categorical_features, random_state=0)
balancer_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), over_sampling)
X_resampled, y_resampled = balancer_pipeline.fit_resample(X, y)

under_sampling = NEW_ENN(categorical_features)
X_resampled, y_resampled = under_sampling.fit_resample(X_resampled, y_resampled)

print(sorted(Counter(y_resampled).items()))

new_X = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_name])], axis=1)
new_X.to_csv('/Users/rlopez/D3M/others/learningData.csv', index_label=index_name)
print(new_X)

