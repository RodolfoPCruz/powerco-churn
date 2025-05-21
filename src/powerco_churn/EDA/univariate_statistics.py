"""
Module to calculate univariate statistics of features in a pandas dataframe.

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




def univariate_statistics(df: pd.DataFrame, round_to: int = 3) -> pd.DataFrame:
    """
    Function to calculate import statistis for univariate analysis

    Args:
        df (pd.DataFrame): input data
            round (int) : number of decimals to be returned
    Returns
        output_df: pd.DataFrame containing the calculated statistics.
        The features will be the index of the dataframe and the
        statitics the columns.

        The univariate statistics in the output dataframe are
        the following:

            type  : datatype of the feature
            count  : number of not nan values
            missing : number of missing values
            unique : number of unique values
            mode : most frequent value
            min_value : min value
            q_1 : first quartile
            median : median
            q_3 : third quartile
            max_value : max value
            mean : mean value
            std : standard deviation
            skew : skew
            kurtosis : kurtosis
    """

    output_df = pd.DataFrame(
        columns=[
            "feature",
            "type",
            "count",
            "missing",
            "unique",
            "mode",
            "min_value",
            "q_1",
            "median",
            "q_3",
            "max_value",
            "mean",
            "std",
            "skew",
            "kurtosis",
        ]
    )
    output_df.set_index("feature", inplace=True)

    for col in df.columns:
        # metrics that be calculated for all data types
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode_series = df[col].mode()
        mode = mode_series[0] if not mode_series.empty else "-"

        if pd.api.types.is_numeric_dtype(df[col]):
            # metrics that only apply to numeric columns
            min_value = round(df[col].min(), round_to)
            q_1 = df[col].quantile(0.25)
            median = df[col].median()
            q_3 = df[col].quantile(0.75)
            max_value = df[col].max()
            mean_ = round(df[col].mean(), round_to)
            std_ = round(df[col].std(), round_to)
            skew = round(df[col].skew(), round_to)
            kurtosis = round(df[col].kurtosis(), round_to)

        else:
            min_value = "-"
            q_1 = "-"
            median = "-"
            q_3 = "-"
            max_value = "-"
            mean_ = "-"
            std_ = "-"
            skew = "-"
            kurtosis = "-"

        output_df.loc[col] = [
            dtype,
            count,
            missing,
            unique,
            mode,
            min_value,
            q_1,
            median,
            q_3,
            max_value,
            mean_,
            std_,
            skew,
            kurtosis,
        ]

    return output_df


def plot_histograms_countplots(
    df: pd.DataFrame, list_of_features: list = None
) -> None:
    """
    The function plots histogram for numeric features and
    countplots for categorical features.

    Args:
        df (pd.DataFrame): data to be plotted
        list_of_feature: features to be plotted. If None,
        all numerical and categorical features in df will be plotted
    """

    if list_of_features is None:
        list_of_features = df.columns

    for col in list_of_features:
        if col not in df:
            print(f"The feature {col} was not in df and was therefore ignored")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.title(col)
            sns.histplot(df[col], stat="density")
        else:
            plt.title(col)
            sns.countplot(data=df, x=df[col])
            if df[col].nunique() > 10:
                plt.xticks(rotation=90)
        plt.show()



