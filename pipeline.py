import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import kagglehub
import os

from transformers import LabelEncoderTransformer
import mlflow

experiment_name = "house_pricing"

mlflow.set_tracking_uri("http://localhost:5001")

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)


def load_dataset():
    path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")
    return pd.read_csv(os.path.join(path, "AmesHousing.csv"))


def _split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def _apply_numerical_transformer(X, X_train, X_test):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    X_train[numeric_features] = numeric_transformer.fit_transform(
        X_train[numeric_features]
    )
    X_test[numeric_features] = numeric_transformer.transform(X_test[numeric_features])

    file_name = "numeric_transformer.pkl"
    pickle.dump(numeric_transformer, open(file_name, "wb"))
    mlflow.log_artifact(local_path=file_name, artifact_path=file_name)

    return X_train, X_test


def _apply_categorical_transformer(X, X_train, X_test):
    categorical_features = X.select_dtypes(include=["object"]).columns

    categorical_transformer = Pipeline(
        steps=[("labelencoder", LabelEncoderTransformer())]
    )

    categorical_transformer.fit(X[categorical_features])
    X_train[categorical_features] = categorical_transformer.transform(
        X_train[categorical_features]
    )
    X_test[categorical_features] = categorical_transformer.transform(
        X_test[categorical_features]
    )

    file_name = "categorical_transformer.pkl"
    pickle.dump(categorical_transformer, open(file_name, "wb"))
    mlflow.log_artifact(local_path=file_name, artifact_path=file_name)

    return X_train, X_test


def feature_engineering(df):
    columns_to_use = [
        "Overall Qual",
        "Exter Qual",
        "Bsmt Qual",
        "Total Bsmt SF",
        "1st Flr SF",
        "Gr Liv Area",
        "Kitchen Qual",
        "Garage Cars",
        "Garage Area",
    ]
    target_column = "SalePrice"

    X = df[columns_to_use]
    y = df[target_column]

    X_train, X_test, y_train, y_test = _split_dataset(X, y)
    X_train, X_test = _apply_categorical_transformer(X, X_train, X_test)
    X_train, X_test = _apply_numerical_transformer(X, X_train, X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("-" * 50)

    # Cross-validation score
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    final_score = cv_score.mean()
    print(f"  Cross-Validation MSE (5-fold): {final_score:.4f}")
    print("=" * 50)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_params(model.get_params())

    from mlflow.models import infer_signature

    signature = infer_signature(X_train)
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

    return final_score


if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test = feature_engineering(df=df)
    score = train_model(X_train, X_test, y_train, y_test)
    print(score)
