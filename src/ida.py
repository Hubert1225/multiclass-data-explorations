import pandas as pd
from scipy.stats import chisquare


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


def missing_values_chisquare_test(column: pd.Series, classes: pd.Series) -> float:
    """Checks if classes distribution among samples
    where value in given column is missing is the same as
    classes distribution in entire data.

    The chi-square test is performed, which tests the null hypothesis that
    the respective classes have the given expected frequencies among missing data,
    where expected frequencies are frequencies in entire data.

    Args:
        column: values of column for consecutive samples
        classes: class labels for consequtive samples

    Returns:
        float: the pvalue of the chi-square test

    """
    missing_classes = classes[column.isna()]
    class_labels = pd.Series(classes.unique())
    # calculate expected freqs of classes among missing values
    classes_dist_all = class_labels.map(
        lambda x: (classes == x).sum() / classes.size
    ).to_numpy()
    classes_freq_expected = classes_dist_all * missing_classes.size
    # calculate observed freqs of classes among missing values
    classes_freq_missing = class_labels.map(
        lambda x: (missing_classes == x).sum()
    ).to_numpy()
    return chisquare(f_obs=classes_freq_missing, f_exp=classes_freq_expected).pvalue
