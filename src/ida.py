import pandas as pd
from scipy.stats import chisquare, shapiro

from randomness import get_random_generator


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


def missing_values_chisquare_test(column: pd.Series, classes: pd.Series) -> float | str:
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
        float | str: the pvalue of the chi-square test if there are missing values
            or 'NA' string otherwise

    """
    missing_classes = classes[column.isna()]
    class_labels = pd.Series(classes.unique())
    # if there are no missing values, return 'NA'
    if missing_classes.size == 0:
        return "NA"
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


def classes_normality_test(column: pd.Series, classes: pd.Series) -> dict[int, float]:
    """Performs statistical test for each class that column values
    of samples from this class are normally distributed.

    For each class, column values are retrieved and then
    the Shapiro-Wilk test for normality is performed.

    Args:
        column: values of column for consecutive samples
        classes: class labels for consequtive samples

    Returns:
        dict[int, float]: dict in which keys are class labels
            and values are corresponding p-values from the Shapiro-Wilk test

    """
    classes_pvalues: dict[int, float] = {}
    for cl in classes.unique():
        class_samples = column[classes == cl]
        classes_pvalues[cl] = shapiro(class_samples).pvalue
    return classes_pvalues


def describe_column(column: pd.Series, classes: pd.Series, dataset_name: str) -> str:
    """Given a data column and corresponding class labels,
    builds a description string for the column.

    Args:
        column: values of column for consecutive samples
        classes: class labels for consequtive samples
        dataset_name: name of the dataset

    Returns:
        str: multiline description string containing information about
            values of the column with taking into account the class labels

    """
    rng = get_random_generator(dataset_name)
    desc = ""
    desc += f"\nDtype: {column.dtype}"
    desc += (
        f"\nRandomly chosen exemplary vals:\n{column[rng.choice(column.size, size=10)]}"
    )
    desc += f"\nNumber of unique values: {column.unique().size}"
    desc += '\n'
    desc += f"\nMinimum: {column.min()}"
    desc += f"\n10. centile: {column.quantile(q=0.1)}"
    desc += f"\n1. quartile: {column.quantile(q=0.25)}"
    desc += f"\nMedian: {column.median()}"
    desc += f"\n3. quartile: {column.quantile(q=0.75)}"
    desc += f"\n90. centile: {column.quantile(q=0.9)}"
    desc += f"\nMaximum: {column.max()}"
    desc += '\n'
    desc += f"\nMissing values percentage: {missing_values_percent(column):.3f} %"
    desc += f"\nChi-square test p-value: {missing_values_chisquare_test(column=column, classes=classes)}"
    desc += '\n'
    desc += f"\nShapiro-Wilk test p-values for classes:"
    classes_pvalues = classes_normality_test(column=column, classes=classes)
    for cl, pvalue in classes_pvalues.items():
        desc += f"\n\tclass {cl}: {pvalue:.3f}"
    return desc
