"""
Module for basic data wrangling
"""

import logging

import pandas as pd

from powerco_churn.utils.logger_utils import configure_logging

configure_logging(log_file_name="wrangling.log")
# logger = logging.getLogger(__name__)


def basic_wrangling(
    df: pd.DataFrame,
    columns: list = None,
    proportion_nan_thresh: float = 0.95,
    proportion_unique_thresh: float = 0.95,
    verbose: bool = True,
):
    """
    Function to execute some basic data wrangling in pandas dataframe.
    The following operations will be executed:

        1. Remove empty or almost empty columns
        2. Remove categorical columns with more than proprotion_unique_thresh
            unique values
        3. Remove columns composed by a single value

    Args:
        df - pandas Dataframe
        columns - list of columns to be wrngled. If None, operations will
                  be applied to all columns in dataframe
        proportion_nan_thresh - If the proportion of NAN values in a column
                    is greater than proportion_nan_thresh the column will be
                    removed. Default = 0.95
        proportion_unique_thresh - If the proportion of unique values in a
                    categorical column is greater than
                    proportion_unique_thresh
                    the column will be removed. Default = 0.95

    Returns:
        df - wrangled pandas dataframe
    """

    df = df.copy()
    if verbose:
        logging.info("Starting data wrangling...")

    if columns is None:
        columns = df.columns

    n_rows = len(df)  # number of rows

    for feature in columns:
        if feature in df.columns:
            num_empty_values = df[feature].isna().sum()
            proportion_nan_values = num_empty_values / n_rows
            num_unique = df[feature].nunique()
            proportion_unique_values = num_unique / n_rows
            data_type = df[feature].dtype
            # remove empty or almost empy columns
            if proportion_nan_values > proportion_nan_thresh:
                df = df.drop(columns=feature)
                if verbose:
                    logging.info(
                        "Feature %s removed. %.2f%% of missing values.",
                        feature,
                        100 * proportion_nan_values,
                    )

            # remove features with only one unique value
            elif num_unique == 1:
                df = df.drop(columns=feature)

                if verbose:
                    logging.info(
                        "Feature %s removed. There is only one  unique value "
                        "in the column.",
                        feature,
                    )

            # Remove categorical columns with more than p
            # roprotion_unique_thresh unique values
            elif (
                proportion_unique_values > proportion_unique_thresh
                and not pd.api.types.is_numeric_dtype(df[feature])
            ):
                df = df.drop(columns=feature)
                if verbose:
                    logging.info(
                        "Feature %s removed. The proportion of unique "
                        "values in "
                        "the feature of "
                        "type %s is %.2f%%",
                        feature,
                        data_type,
                        100 * proportion_unique_values,
                    )

        else:
            if verbose:
                logging.info(
                    "The feature %s is not in the dataframe.", feature
                )

    return df


if __name__ == "__main__":
    nba = load_csv_from_data("nba/nba_salaries.csv")
    nba_wrangled = basic_wrangling(nba)
    print(nba_wrangled.head())
