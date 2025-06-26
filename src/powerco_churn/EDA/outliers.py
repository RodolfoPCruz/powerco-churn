"""
Module to detect an treat outliers in a pandas dataframe
The function clean_outiliers uses the empirical rule and Tukey rule to detect
outliers.
The function clean_outliers_using_dbscan uses the clustering algorrithm DBSCAN
to detect outliers
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.cluster import DBSCAN
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

from powerco_churn.utils.logger_utils import configure_logging

configure_logging(log_file_name="outliers.log")

sns.set_style("darkgrid")


def calculate_outlier_threshold(
    df: pd.DataFrame, feature: str, skew_thresold: float = 1
) -> tuple[float, float]:
    """
    Calculate the inferior and superior thresholds for outlier detection.
    Values above max_thresh and values below min_thresh are considered
    outliers

    Args:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The name of the feature to be processed.
        skew_thresold (float): The skewness threshold used to determine
        whether the distribution is normal or not.

    Returns:
        tuple: A tuple containing the inferior and superior thresholds
        for outlier detection.
    """
    skew = df[feature].skew()
    if skew > -1 * skew_thresold and skew < skew_thresold:
        # empirical rule to detect outliers in normal distributions
        min_thresh = df[feature].mean() - 3 * df[feature].std()
        max_thresh = df[feature].mean() + 3 * df[feature].std()
    else:
        # apply the Tukey rule to detect outliers in distributions
        # that are not normal
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1  # interquartile range
        max_thresh = q3 + 1.5 * (iqr)
        min_thresh = q1 - 1.5 * (iqr)

    return (min_thresh, max_thresh)


def clean_outliers(
    df: pd.DataFrame,
    features_list: list = None,
    outlier_treatment: str = None,
    output_column: str = None,
    skew_thresold: float = 1,
    max_iter: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Detect and treat outliers. Two criteria will be used to detect outliers:
        - empirical rule to detect outliers in normal distributions;
        - Tukey rule to detect outliers in distributions that are not normal.
    A distribution will be considered normal when its skewness
    is between -1 * skew_thresold and skew_thresold.One of the two criteria
    will be used to calculate max_thresh and min_thresh. Values above
    max_thresh and values below min_thresh are considered outliers.


    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned.
    features_list : list, optional
        The list of features to be cleaned, by default, is None. If None,
        the function will clean all features.
    outlier_treatment : str, optional
        The treatment to be applied to outliers, by default, None.
        Can be 'remove'; 'replace'; 'impute', or None:
            - remove: rows containing outliers are completely removed;
            - replace: outliers above max_thresh and outliers below
              min_thresh are replaced by max_thresh and min_thresh,
              respectively;
            - impute: outliers above max_thresh and outliers below min_thresh
              are imputed using the Iterative imputer method form
              scikit-learn;
            - None: no treatment is applied to outliers.
    output_column : str, optional
        The name of the output column, by default is None. The output column
        will not be used when calculating the values to be imputed.
        If None, it is considered that the output column is not in df.
    skew_thresold : float, optional
        The skewness threshold for a distribution to be considered normal,
        by default 1.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.

    """
    set_config(transform_output="pandas")

    if verbose:
        logging.info(
        "Detecting and treating outliers " "using traditional methods..."
        )

    df = df.copy()
    initial_number_rows = df.shape[0]
    df = df.dropna()
    final_number_rows = df.shape[0]
    removed_rows = initial_number_rows - final_number_rows
    if verbose:
        logging.info(
            f"{removed_rows} rows were removed because "
            "they contained NaN values."
        )

    outliers_thresholds = {}

    if features_list is None:
        features_list = df.columns.to_list()

    if outlier_treatment not in ["remove", "replace", "impute", None]:
        if verbose:
            logging.info(
                f"The outlier_treatment {outlier_treatment} is not "
                "valid. No treatment will be applied to outliers."
            )
        outlier_treatment = None

    # remove output_column from features_list. The output column can not be
    # used when calculating the values to be imputed
    if output_column is not None and output_column in features_list:
        features_list_impute_method = features_list.copy()
        features_list_impute_method.remove(output_column)
    else:
        features_list_impute_method = features_list

    for feature in features_list:
        # the input features can not be used to impute values to output feature
        if feature == output_column and outlier_treatment == "impute":
            continue
        # test whether the feature is in the dataframe
        if feature in df.columns:
            # only numeric columns will be cleaned
            if pd.api.types.is_numeric_dtype(df[feature]):

                # test whether the feature has only one unique value
                if df[feature].nunique() == 1:
                    if verbose:
                        logging.info(
                            f"The feature {feature} has only one "
                            "unique value and was therefore ignored."
                        )

                # test whether the feature is binary (only true or
                # false values)
                elif set(df[feature].dropna().unique()).issubset(
                    {0, 1}
                ) or set(df[feature].dropna().unique()).issubset(
                    {True, False}
                ):
                    if verbose:
                        logging.info(
                            f"The feature {feature} is binary "
                            "(only true or false values) and was "
                            "therefore ignored."
                        )

                # look for outliers
                else:
                    # calculate max_thresh and min_thresh

                    min_thresh, max_thresh = calculate_outlier_threshold(
                        df, feature, skew_thresold
                    )

                    outliers_thresholds[feature] = (min_thresh, max_thresh)  

                    # values above max_thresh and values below min_thresh
                    # are considered outliers
                    count_max_outlier = len(df.loc[df[feature] > max_thresh])
                    count_min_outlier = len(df.loc[df[feature] < min_thresh])
                    if verbose:
                        logging.info(
                            f"The feature {feature} has "
                            f"{count_max_outlier} "
                            f"values above {max_thresh}"
                        )
                    if verbose:
                        logging.info(
                            f"The feature {feature} "
                            f"has {count_min_outlier} "
                            f"values below {min_thresh}"
                        )

                    has_outliers = (
                        count_max_outlier > 0 or count_min_outlier > 0
                    )

                    if not has_outliers:
                        continue

                    if outlier_treatment == "remove":
                        df = df[
                            (df[feature] >= min_thresh)
                            & (df[feature] <= max_thresh)
                        ]
                        logging.info(f"{feature}: outliers removed")

                    elif outlier_treatment == "replace":
                        if not pd.api.types.is_float_dtype(df[feature]):
                            df[feature] = df[feature].astype(float)
                        logging.info(f"{feature}: outliers replaced")
                        df.loc[df[feature] > max_thresh, feature] = max_thresh
                        df.loc[df[feature] < min_thresh, feature] = min_thresh

            else:
                if verbose:
                    logging.info(
                        f"The feature {feature} is not "
                        "numeric and was therefore ignored"
                    )

        else:
            if verbose:
                logging.info(
                    f"The {feature} was not in the dataframe."
                )
    if outlier_treatment == 'impute':
        df_temp = df[features_list_impute_method].copy()
        for feature in outliers_thresholds:
            df_temp.loc[df_temp[feature] > outliers_thresholds[feature][1],
                                                        feature] = np.nan
            
            df_temp.loc[df_temp[feature] < outliers_thresholds[feature][0], 
                                                        feature] = np.nan
        
        num_outliers = df_temp.isna().sum().sum() 
        logging.info(
                    f"{num_outliers} OUTLIERS."
                ) 
        if num_outliers > 0:
            df_temp = pd.get_dummies(df_temp, drop_first = True)
            imputer = IterativeImputer(max_iter = max_iter, 
                        random_state = random_state)
            df_temp = imputer.fit_transform(df_temp)
            cleaned_features = list(outliers_thresholds.keys())
            df[cleaned_features] = df_temp[cleaned_features]
            if df_temp.isna().sum().sum() == 0:
                logging.info(
                    f"{num_outliers} imputed."
                )   

    return df


def search_eps_dbscan(
    df: pd.DataFrame,
    distance_metric: str = "manhattan",
    min_samples: int = 5,
    eps_step: float = 0.01,
    desired_percentage_outliers: float = 0.02,
    max_eps: float = 5.0,
    plot: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Find the optimal eps value for DBSCAN that results in a desired
    percentage of outliers.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - distance_metric (str): Distance metric for DBSCAN.
    - min_samples (int): Minimum samples for core point in DBSCAN.
    - eps_step (float): Step size for increasing eps.
    - desired_percentage_outliers (float): Target outlier percentage (0-1).
    - max_eps (float): Upper limit for eps search.
    - plot (bool): Whether to plot eps vs. outlier percentage.
    - verbose (bool): Whether to log cleaning info.

    Returns:
    - tuple[pd.DataFrame, float]: DataFrame of eps vs. outlier percentages,
    and selected eps value.
    """
    if verbose:
        logging.info("Starting the search for the best eps value")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df_temp = df.dropna(how="all", axis=1).dropna(how="any", axis=0)

    if verbose:
        removed_rows = df.shape[0] - df_temp.shape[0]
        removed_cols = df.shape[1] - df_temp.shape[1]
        logging.info(
            f"{removed_rows} rows and {removed_cols} columns "
            "removed due to missing values."
        )

    # Encode categorical variables
    df_temp = pd.get_dummies(df_temp, drop_first=True)

    if df_temp.empty:
        raise ValueError("DataFrame became empty after encoding.")

    # Scale features
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_temp)

    # Store eps and outlier percentages
    eps_values = []
    outlier_percentages = []

    for eps in np.arange(eps_step, max_eps + eps_step, eps_step):
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric=distance_metric
        )
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric=distance_metric
        )
        labels = dbscan.fit_predict(df_scaled)
        num_outliers = np.count_nonzero(labels == -1)
        percentage = round(100 * num_outliers / len(df_scaled), 2)

        eps_values.append(eps)
        outlier_percentages.append(percentage)

        # Stop early if no more outliers
        if num_outliers == 0:
            break

    results_df = pd.DataFrame(
        {"eps": eps_values, "percentage_outliers(%)": outlier_percentages}
    )
    results_df = pd.DataFrame(
        {"eps": eps_values, "percentage_outliers(%)": outlier_percentages}
    )

    results_df["diff"] = np.abs(
        results_df["percentage_outliers(%)"]
        - 100 * desired_percentage_outliers
    )
    results_df["diff"] = np.abs(
        results_df["percentage_outliers(%)"]
        - 100 * desired_percentage_outliers
    )
    best_idx = results_df["diff"].idxmin()
    best_eps = round(results_df.loc[best_idx, "eps"], 4)
    best_pct = results_df.loc[best_idx, "percentage_outliers(%)"]

    if verbose:
        logging.info(f"Best eps: {best_eps} for ~{best_pct}% outliers")

    # Plot if requested
    if plot:
        sns.lineplot(data=results_df, x="eps", y="percentage_outliers(%)")
        plt.scatter(best_eps, best_pct, color="red")
        plt.annotate(
            f"eps = {best_eps}\n% outliers = {best_pct}",
            xy=(best_eps, best_pct),
            xytext=(best_eps + 0.1, best_pct + 0.1),
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
        sns.lineplot(data=results_df, x="eps", y="percentage_outliers(%)")
        plt.scatter(best_eps, best_pct, color="red")
        plt.annotate(
            f"eps = {best_eps}\n% outliers = {best_pct}",
            xy=(best_eps, best_pct),
            xytext=(best_eps + 0.1, best_pct + 0.1),
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
        plt.title("DBSCAN eps vs. Percentage of Outliers")
        plt.grid(True)
        plt.show()

    return results_df, best_eps


def clean_outliers_using_dbscan(
    df: pd.DataFrame,
    eps: float,
    distance_metric: str = "manhattan",
    min_samples: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove outliers using DBSCAN clustering.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    distance_metric : str, optional
        Distance metric to use for DBSCAN. Default is 'manhattan'.
    min_samples : int, optional
        Minimum number of points to form a dense region. Default is 5.
    verbose : bool, optional
        If True, prints detailed logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.
    """
    df_clean = df.copy()

    # Drop empty columns and rows with NaNs
    initial_shape = df_clean.shape
    df_temp = df_clean.dropna(axis="columns", how="all").dropna(
        axis="rows", how="any"
    )
    cleaned_shape = df_temp.shape

    if verbose:
        logging.info(
            f"Removed {initial_shape[0] - cleaned_shape[0]} rows and "
            f"{initial_shape[1] - cleaned_shape[1]} "
            "columns with missing values."
        )

    # Save original indices
    original_indices = df_temp.index

    # Encode categoricals and scale
    df_temp = pd.get_dummies(df_temp, drop_first=True)
    scaled = MinMaxScaler().fit_transform(df_temp)

    # DBSCAN clustering
    db = DBSCAN(
        eps=eps, min_samples=min_samples, metric=distance_metric, n_jobs=-1
    )
    labels = db.fit_predict(scaled)

    # Identify outliers
    outlier_indices = original_indices[labels == -1]
    if verbose:
        logging.info(
            f"Removing {len(outlier_indices)} outliers "
            f"({round(len(outlier_indices) / len(df_clean) * 100, 2)}%)"
        )

    # Drop outliers
    return df_clean.drop(index=outlier_indices)


if __name__ == "__main__":
    # nba = load_csv_from_data("nba/nba_salaries.csv")
    # nba_cleaned = clean_outliers(nba)
    # nba = load_csv_from_data("nba/nba_salaries.csv")
    # nba_cleaned = clean_outliers(nba)
    insurance = load_csv_from_data("insurance/insurance.csv")
    insurance_cleaned = clean_outliers(insurance, outlier_treatment = 'impute',
                            max_iter = 1000, verbose = True)
    #results_df, best_eps = search_eps_dbscan(
    #    insurance, plot=False, verbose=False
    #)
    #insurance_cleaned = clean_outliers_using_dbscan(insurance, best_eps)
