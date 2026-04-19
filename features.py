import pandas as pd
import numpy as np

FEATURE_NAMES = ['year', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']


def load_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df[FEATURE_NAMES]


def get_feature_names() -> list:
    return FEATURE_NAMES.copy()


class FeatureEngineer:
    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        self.fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        return X[self.feature_names]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)