import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from params import params
from ida import missing_values_percent


set_config(transform_output="pandas")


def get_column_dropper(columns_to_drop: list[str]) -> ColumnTransformer:
    """Returns transformer object that drops some subset of
    columns
    """
    return ColumnTransformer(
        transformers=[
            ("column_dropper", "drop", columns_to_drop),
        ]
    )


def get_columns_to_drop(X: pd.DataFrame, dataset_name: str) -> list[str]:
    max_missing_percent = params[dataset_name].get("max_missing_percent")
    columns_to_drop = []
    for column in X.columns:
        if missing_values_percent(X[column]) > max_missing_percent:
            columns_to_drop.append(column)
    return columns_to_drop


def get_missing_values_pipeline(X: pd.DataFrame, dataset_name: str) -> Pipeline:
    columns_to_drop = get_columns_to_drop(X=X, dataset_name=dataset_name)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("drop_columns", get_column_dropper(columns_to_drop)),
        ]
    )
