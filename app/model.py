import mlflow
import os
import pandas as pd


class Model:
    def __init__(self):
        mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:5001")
        self._model_name_prefix = os.getenv("MODEL_NAME", "house_pricing")
        self._model_version = os.getenv("MODEL_VERSION", "1")

        mlflow.set_tracking_uri(mlflow_uri)

        self._model = self.get_model()
        self._numerical_transformer = self.get_numerical_transformer()
        self._categorical_transformer = self.get_categorical_transformer()

        self._categorical_features = ['Exter Qual', 'Bsmt Qual', 'Kitchen Qual']
        self._numerical_features = ['Overall Qual', 'Total Bsmt SF', '1st Flr SF', 'Gr Liv Area', 'Garage Cars', 'Garage Area']

    def _get_sklearn_model(self, name):
        model_uri = f"models:/{self._model_name_prefix}_{name}/{self._model_version}"
        model = mlflow.sklearn.load_model(model_uri)

        return model

    def get_model(self):
        return self._get_sklearn_model(name="model")

    def get_numerical_transformer(self):
        return self._get_sklearn_model(name="numerical_transformer")

    def get_categorical_transformer(self):
        return self._get_sklearn_model(name="categorical_transformer")

    def predict(self, data: pd.DataFrame):
        data[self._categorical_features] = self._categorical_transformer.transform(data[self._categorical_features])
        data[self._numerical_features] = self._numerical_transformer.transform(data[self._numerical_features])
        return self._model.predict(data)
