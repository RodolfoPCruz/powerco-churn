"""
Module to calculate and crete plots of
bivariate statistics of features in a pandas dataframe.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

#from eda_toolkit.utils.data_loader import load_csv_from_data

# pylint: disable=too-many-arguments


def compute_regression_statistics(
    df: pd.DataFrame, column_1: str, column_2: str, round_to: int = 3
) -> dict:
    """
    Compute regression and correlation statistics between two numeric
    columns.

    Returns:
        dict: Dictionary containing regression slope, intercept, p-values,
        correlations, and skewness.
    """
    result_lin_regression = stats.linregress(df[column_1], df[column_2])
    res_spearman = stats.spearmanr(df[column_1], df[column_2])
    res_kendall = stats.kendalltau(df[column_1], df[column_2])

    return {
        "slope": round(result_lin_regression.slope, round_to),
        "intercept": round(result_lin_regression.intercept, round_to),
        "pearson_r": round(result_lin_regression.rvalue, round_to),
        "pearson_p": round(result_lin_regression.pvalue, round_to),
        "spearman": round(res_spearman.statistic, round_to),
        "spearman_p": round(res_spearman.pvalue, round_to),
        "kendall": round(res_kendall.statistic, round_to),
        "kendall_p": round(res_kendall.pvalue, round_to),
        "skew_1": round(df[column_1].skew(), round_to),
        "skew_2": round(df[column_2].skew(), round_to),
    }


def generate_regression_plot(
    df: pd.DataFrame, column_1: str, column_2: str, round_to: int = 3
):
    """
    Generate a linear regression plot between two numeric columns
        in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input data.
        column_1 (str): Column name for the X-axis.
        column_2 (str): Column name for the Y-axis.
        round_to (int): Decimal precision for statistical results.

    Returns:
        None
    """
    df = df[[column_1, column_2]].dropna()

    if not (
        pd.api.types.is_numeric_dtype(df[column_1])
        and pd.api.types.is_numeric_dtype(df[column_2])
    ):
        raise TypeError("Both columns must be numeric.")

    stats_summary = compute_regression_statistics(
        df, column_1, column_2, round_to
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.regplot(
        data=df,
        x=column_1,
        y=column_2,
        ax=ax1,
        line_kws={"color": "darkorange"},
    )
    ax1.set_xlabel(column_1)
    ax1.set_ylabel(column_2)
    ax1.set_title(f"Regression plot: {column_1} vs {column_2}")

    # Text summary
    text_str = (
        f"y = {stats_summary['slope']}x + {stats_summary['intercept']}\n"
        f"Pearson r = {stats_summary['pearson_r']}, "
        f"p = {stats_summary['pearson_p']}\n"
        f"Spearman ρ = {stats_summary['spearman']}, "
        f"p = {stats_summary['spearman_p']}\n"
        f"Kendall τ = {stats_summary['kendall']}, "
        f"p = {stats_summary['kendall_p']}\n"
        f"Skewness {column_1} = {stats_summary['skew_1']}\n"
        f"Skewness {column_2} = {stats_summary['skew_2']}"
    )
    ax2.text(0.5, 0.5, text_str, fontsize=12, ha="center", va="center")
    ax2.axis("off")  # Hide the axes
    ax2.set_title("Statistics")

    plt.tight_layout()
    plt.show()

    return fig, ax1, ax2


def generate_bar_plot(
    df: pd.DataFrame,
    numerical_feature: str,
    categorical_feature: str,
    round_to: int = 3,
    alpha: float = 0.05,
    plot_type: str = "bar",
):
    """
    Plots a bar or violin plot between a numerical and a categorical feature,
    and performs statistical tests (t-test or ANOVA + pairwise t-tests).

    Parameters:
    - df: DataFrame
    - numerical_feature: column name of numeric variable
    - categorical_feature: column name of categorical variable
    - round_to: number of decimal places in output
    - alpha: significance level
    - plot_type: "bar" or "violin"
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Choose plot type
    if plot_type == "bar":
        sns.barplot(
            data=df, x=categorical_feature, y=numerical_feature, ax=ax1
        )
    elif plot_type == "violin":
        sns.violinplot(
            data=df,
            x=categorical_feature,
            y=numerical_feature,
            ax=ax1,
            inner="box",
        )
    else:
        raise ValueError("Invalid plot_type. Choose 'bar' or 'violin'.")

    # Grouping and filtering
    groups_list = []
    valid_groups = []

    for group in df[categorical_feature].unique():
        temp_group = df[df[categorical_feature] == group][numerical_feature]
        if len(temp_group) >= 2:
            groups_list.append(temp_group)
            valid_groups.append(group)

    num_groups = len(groups_list)

    if num_groups < 2:
        raise ValueError(
            "It is necessary to have at least two valid groups in "
            "the categorical feature."
        )

    # Statistical tests
    if num_groups == 2:
        t_stat, p_value = stats.ttest_ind(*groups_list, equal_var=False)
        text_str = f"t-test:\nt={round(t_stat, round_to)}, "
        text_str += f"p={round(p_value, round_to)}"
    else:
        f_stat, p_val_anova = stats.f_oneway(*groups_list)
        text_str = f"ANOVA:\nF={round(f_stat, round_to)}, "
        text_str += f"p={round(p_val_anova, round_to)}\n"

        pairs = list(combinations(valid_groups, 2))
        bonferroni = round(alpha / len(pairs), round_to)
        text_str += f"\nPairwise t-tests (Bonf α={bonferroni}):\n"
        for a, b in pairs:
            vals_a = df[df[categorical_feature] == a][numerical_feature]
            vals_b = df[df[categorical_feature] == b][numerical_feature]
            t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            text_str += f"{a} vs {b}: t={round(t_stat, round_to)}, "
            text_str += f"p={round(p_value, round_to)}\n"

    if num_groups > 10:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    ax2.text(
        0.5,
        0.5,
        text_str.strip(),
        fontsize=12,
        ha="center",
        va="center",
        wrap=True,
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2


def generate_heat_map(
    df: pd.DataFrame,
    categorical_feat_1: str,
    categorical_feat_2: str,
    axis_sum: int = None,
    round_to: int = 3,
):  # pylint: disable=too-many-arguments
    """
    Generates a heatmap between two categorical features in a DataFrame,
    along with Chi-squared statistical test results.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame.
        categorical_feat_1 (str): Name of the first categorical feature.
        categorical_feat_2 (str): Name of the second categorical feature.
        axis_sum (int or None): Determines how proportions are calculated:
            - None: Global proportion (relative to all values).
            - 0: Column-wise (relative to column totals).
            - 1: Row-wise (relative to row totals).
        round_to (int): Number of decimal places to round numerical outputs.

    Returns:
        tuple: (fig, ax1, ax2) — the matplotlib Figure and Axes objects.
    """
    if axis_sum not in (None, 0, 1):
        raise ValueError("axis_sum must be one of: None, 0, or 1.")

    # Build contingency table
    contingency_table = pd.crosstab(
        df[categorical_feat_1], df[categorical_feat_2]
    )

    # Chi-squared test
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    chi2 = round(chi2, round_to)
    p_value = round(p_value, round_to)

    text_str = (
        "Chi-squared test:\n"
        f"Chi-squared statistic: {chi2}\n"
        f"p-value: {p_value}\n"
    )

    # Prepare annotation labels
    values = contingency_table.values
    shape = values.shape
    labels = np.empty(shape, dtype=object)

    if axis_sum is None:
        total = values.sum()
        for i in range(shape[0]):
            for j in range(shape[1]):
                count = values[i, j]
                percent = count / total * 100
                labels[i, j] = f"{count}\n({percent:.1f}%)"
    else:
        totals = values.sum(axis=axis_sum)
        for i in range(shape[0]):
            for j in range(shape[1]):
                count = values[i, j]
                if axis_sum == 0:
                    percent = count / totals[j] * 100
                else:  # axis_sum == 1
                    percent = count / totals[i] * 100
                labels[i, j] = f"{count}\n({percent:.1f}%)"

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(
        contingency_table,
        annot=labels,
        fmt="",
        cmap="YlGnBu",
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_title("Contingency Heatmap")

    ax2.text(0.5, 0.5, text_str, fontsize=12, ha="center", va="center")
    ax2.axis("off")
    ax2.set_title("Statistics")

    plt.tight_layout()
    plt.show()

    return fig, ax1, ax2


if __name__ == "__main__":
    insurance = load_csv_from_data("insurance/insurance.csv")
    generate_regression_plot(insurance, "bmi", "charges")
    _, _, _ = generate_bar_plot(insurance, "charges", "region")
    _, _, _ = generate_heat_map(insurance, "sex", "smoker")
