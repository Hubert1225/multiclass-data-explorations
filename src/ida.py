import pandas as pd


def missing_values_percent(column: pd.Series) -> float:
    """Returns percentage of missing values
    in the values of a column

    Args:
        column: column values

    Returns:
        float: value from range [0.0, 100.0] being
            the percentage of missing values

    """
    return column.isna().sum() * 100.0 / column.size
