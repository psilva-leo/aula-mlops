from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformers = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.transformers[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, le in self.transformers.items():
            X_transformed.loc[:,col] = le.transform(X_transformed[col].astype(str))
        return X_transformed
