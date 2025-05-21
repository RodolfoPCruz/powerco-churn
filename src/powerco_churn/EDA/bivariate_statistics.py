"""
Module to calculate bivariate statistics of features in a pandas dataframe.
"""

from dataclasses import dataclass

# from itertools import combinations
# import matplotlib.pyplot as plt
import pandas as pd

# import seaborn as sns
from scipy import stats


# from eda_toolkit.utils.logger_utils import configure_logging


@dataclass
class StatMeta:
    columns: list
    missing: float
    dtype: str
    unique_vals: int


def bivariate_stats(df: pd.DataFrame, target: str, round_to: int = 3):
    """
    Generate bivariate statistics between each feature and a target column
        in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        target (str): Target column name.
        round_to (int): Decimal precision for rounded values.

    Returns:
        pd.DataFrame: Summary of statistics between each feature and
            the target.
    """
    columns = [
        "missing",
        "type",
        "unique_values",
        "skew",
        "p_value",
        "r",
        "y = mx + b",
        "spearman",
        "spearman_pvalue",
        "kendalltau",
        "kendalltau_pvalue",
        "chi2",
        "ttest",
        "F",
    ]
    output_df = pd.DataFrame(columns=columns)

    for feature in df:
        if feature == target:
            continue

        df_temp = df[[feature, target]].dropna()
        missing = round((len(df) - len(df_temp)) / len(df) * 100, round_to)
        dtype = df_temp[feature].dtype
        unique_vals = df_temp[feature].nunique()
        meta = StatMeta(columns, missing, dtype, unique_vals)

        if len(df_temp) < 2:
            output_df.loc[feature] = result_row(meta)
            continue

        # no variance = no stats
        if unique_vals == 1:
            output_df.loc[feature] = result_row(meta)
            continue

        if pd.api.types.is_numeric_dtype(
            df_temp[feature]
        ) and pd.api.types.is_numeric_dtype(df_temp[target]):
            output_df.loc[feature] = handle_numeric_numeric(
                df_temp, feature, target, round_to, meta
            )
        elif not pd.api.types.is_numeric_dtype(
            df_temp[feature]
        ) and not pd.api.types.is_numeric_dtype(df_temp[target]):
            output_df.loc[feature] = handle_categorical_categorical(
                df_temp, feature, target, round_to, meta
            )
        elif (
            pd.api.types.is_numeric_dtype(df_temp[feature])
            and df_temp[target].nunique() == 2
        ):
            output_df.loc[feature] = handle_ttest(
                df_temp, feature, target, round_to, meta
            )
        elif (
            pd.api.types.is_numeric_dtype(df_temp[target])
            and df_temp[feature].nunique() == 2
        ):
            output_df.loc[feature] = handle_ttest(
                df_temp, target, feature, round_to, meta, flip=True
            )
        elif is_invalid_categorical(df_temp, feature, target):
            output_df.loc[feature] = result_row(meta)
        elif pd.api.types.is_numeric_dtype(df_temp[feature]):
            output_df.loc[feature] = handle_anova(
                df_temp, feature, target, round_to, meta
            )
        else:
            output_df.loc[feature] = handle_anova(
                df_temp, target, feature, round_to, meta
            )

    return output_df


def result_row(meta: StatMeta, **kwargs):
    """
    Build a result row for the output DataFrame with provided statistics.

    Parameters:
        columns (list): All column headers for the output.
        missing (float): Percentage of missing data.
        dtype: Data type of the feature.
        unique_vals (int): Number of unique values.
        **kwargs: Additional statistic key-value pairs.

    Returns:
        dict: Row formatted with all required keys.
    """
    row = {col: "-" for col in meta.columns}
    row.update(
        {
            "missing": f"{meta.missing}%",
            "type": meta.dtype,
            "unique values": meta.unique_vals,
        }
    )
    row.update(kwargs)
    return row


def handle_numeric_numeric(df_temp, feature, target, round_to, meta: StatMeta):
    """
    Compute correlations and regression line for numeric-numeric
        feature-target pairs.
    """
    linreg = stats.linregress(df_temp[feature], df_temp[target])
    spearman = stats.spearmanr(df_temp[feature], df_temp[target])
    kendall = stats.kendalltau(df_temp[feature], df_temp[target])
    return result_row(
        meta,
        skew=round(df_temp[feature].skew(), round_to),
        p_value=round(linreg.pvalue, round_to),
        r=round(linreg.rvalue, round_to),
        **{
            "y = mx + b": (
                f"y = {round(linreg.slope, round_to)}x + "
                f"{round(linreg.intercept, round_to)}"
            )
        },
        spearman=round(spearman.statistic, round_to),
        spearman_pvalue=round(spearman.pvalue, round_to),
        kendalltau=round(kendall.statistic, round_to),
        kendalltau_pvalue=round(kendall.pvalue, round_to),
    )


def handle_categorical_categorical(
    df_temp, feature, target, round_to, meta: StatMeta
):
    """
    Compute Chi-squared test for categorical-categorical
        feature-target pairs.
    """
    chi2, p, _, _ = stats.chi2_contingency(
        pd.crosstab(df_temp[feature], df_temp[target])
    )
    return result_row(
        meta, p_value=round(p, round_to), chi2=round(chi2, round_to)
    )


def handle_ttest(
    df_temp, numeric_col, cat_col, round_to, meta: StatMeta, flip=False
):  # pylint: disable=too-many-arguments
    """
    Compute T-test for numeric-categorical feature-target pairs
        (binary categorical only).

    Parameters:
        flip (bool): Whether the roles of numeric and categorical
            columns are swapped.
    """
    unique = df_temp[cat_col].unique()
    group1 = df_temp[df_temp[cat_col] == unique[0]][numeric_col]
    group2 = df_temp[df_temp[cat_col] == unique[1]][numeric_col]
    tstat, p = stats.ttest_ind(group1, group2)
    skew = round(df_temp[numeric_col].skew(), round_to) if not flip else "-"
    return result_row(
        meta,
        p_value=round(p, round_to),
        ttest=round(tstat, round_to),
        skew=skew,
    )


def handle_anova(df_temp, numeric_col, cat_col, round_to, meta: StatMeta):
    """
    Compute ANOVA for numeric-categorical pairs with more than two groups.
    """
    groups = [
        df_temp[df_temp[cat_col] == cat][numeric_col]
        for cat in df_temp[cat_col].unique()
    ]
    # Keep only groups with at least 2 samples
    valid_groups = [g for g in groups if len(g) >= 2]

    # check the number of valid groups
    if len(valid_groups) < 2:
        return result_row(meta)

    # if number of valid groups is 2, perform t-test
    if len(valid_groups) == 2:
        tstat, p = stats.ttest_ind(valid_groups[0], valid_groups[1])
        return result_row(
            meta,
            p_value=round(p, round_to),
            ttest=round(tstat, round_to),
            skew=round(df_temp[numeric_col].skew(), round_to),
        )

    try:
        # Perform ANOVA
        f, p = stats.f_oneway(*groups)
        return result_row(
            meta,
            p_value=round(p, round_to),
            F=round(f, round_to),
            skew=round(df_temp[numeric_col].skew(), round_to),
        )
    except ValueError as ve:
        # Handle specific ValueError exception
        print(f"ValueError: {ve}")  # Optional logging
        return result_row(meta)
    except TypeError as te:
        # Handle specific TypeError exception
        print(f"TypeError: {te}")  # Optional logging
        return result_row(meta)


def is_invalid_categorical(df_temp, feature, target):
    """
    Check whether both feature and target are non-numeric
    with all unique values.
    """
    return (
        not pd.api.types.is_numeric_dtype(df_temp[feature])
        and df_temp[feature].nunique() == len(df_temp)
    ) or (
        not pd.api.types.is_numeric_dtype(df_temp[target])
        and df_temp[target].nunique() == len(df_temp)
    )


