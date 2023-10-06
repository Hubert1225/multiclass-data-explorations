from sklearn import set_config
from sklearn.compose import ColumnTransformer


set_config(transform_output="pandas")


def get_column_dropper(columns_to_drop: list[str]) -> ColumnTransformer:
    """Returns transformer object that drops some subset of
    columns
    """
    return ColumnTransformer(
        transformers=[
            ('column_dropper', 'drop', columns_to_drop),
        ]
    )
