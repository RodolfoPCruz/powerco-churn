"""
Module to reduce skewness of features in pandas dataframe
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer


from powerco_churn.utils.logger_utils import configure_logging

configure_logging(log_file_name="skweness.log")

sns.set_style("darkgrid")


def remove_nans_and_subsample(
    df: pd.DataFrame,
    feature: str,
    verbose: bool = True,
    subsample_limit: int = 5000,
):
    """
    Removes NaNs from the specified feature and optionally subsamples
    the DataFrame if it has more rows than `subsample_limit`.

    Returns:
        - df_cleaned: DataFrame without NaNs in the feature
        - df_sampled: Possibly subsampled version of df_cleaned (or same as
            df_cleaned)
        - was_subsampled: Boolean flag
    """
    df = df.copy()
    initial_len = len(df)
    df = df.dropna(subset=[feature])

    if verbose:
        message = (
            f"Removed {initial_len - len(df)} rows with NaN in '{feature}"
        )
        logging.info(message)

    if len(df) > subsample_limit:
        df_sampled = df.sample(n=subsample_limit, random_state=42)
        if verbose:
            logging.info(
                f"Subsampled to {len(df_sampled)} rows from {len(df)}"
            )
        return df, df_sampled, True

    if verbose:
        logging.info("No subsampling applied")
    return df, df, False


def calculate_transformations(
    x: pd.DataFrame,
    transformation: str,
    power: float = None,
) -> pd.Series:
    """
    Calculate the transformation based on the specified type and power.
    Parameters:
        x (pd.Series): The feature series to transform.
        transformation (str): The type of transformation ('power', 'log',
             'yeo').
        round_to (int): The number of decimal places to round the skew result.
        power (float): The power to apply for the transformation.
    Returns:
        pd.Series: The transformed series.
    """
    epsilon = 1e-7  # Prevent log(0)

    if transformation == "power":
        transformed = np.power(x, power)
        skew = transformed.skew()
        return transformed, skew.item()
    elif transformation == "log":
        transformed = np.log(x + epsilon)
        skew = transformed.skew()
        return transformed, skew.item()
    elif transformation == "yeo":
        pwr = PowerTransformer(method="yeo-johnson")
        transformed = pwr.fit_transform(x)
        transformed = pd.DataFrame(transformed)
        # transformed = transformed.squeeze()
        skew = transformed.skew()
        return transformed, skew.item()
    else:
        raise ValueError("Invalid transformation type")


def plot_transformations(
    titles_series_dict: dict[str, pd.Series],
    original_series: pd.Series,
    original_title: str,
) -> None:
    """
    Plot original and transformed feature distributions.

    Parameters:
        titles_series_dict (dict): Dictionary mapping titles to transformed
            Series.
        original_series (pd.Series): Original feature data.
        original_title (str): Title for the original distribution plot.
    """
    n = len(titles_series_dict) + 1
    rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    axs = axs.flatten()

    sns.histplot(original_series, ax=axs[0], kde=True, stat="density")
    axs[0].set_title(original_title)

    for i, (title, series) in enumerate(titles_series_dict.items(), start=1):
        sns.histplot(series, ax=axs[i], kde=True, stat="density")
        axs[i].set_title(title)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def apply_transformations(
    df_temp: pd.DataFrame,
    feature: str,
    initial_skew: float,
    max_power: int,
    plot: bool = False,
    round_to: int = 4,
    skew_threshold: float = 0.5,
    num_power_iterations: int = 100,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """
    Apply transformations to reduce skewness in a DataFrame column.

    Parameters:
        df_temp (pd.DataFrame): DataFrame containing the feature to transform.
        feature (str): Column name of the feature.
        initial_skew (float): Initial skewness value.
        max_power (int): Upper limit for inverse power transformation.
        plot (bool): Whether to generate histograms of transformations.
        round_to (int): Number of decimal places to round skewness results.
        skew_threshold (float): Threshold above which skew is considered
            significant.
        num_power_iterations (int): Number of inverse power tests to try.

    Returns:
        transformed (dict): Transformed feature series.
        results (dict): Absolute skewness values of the transformations.
    """
    results = {}
    transformed = {}

    initial_skew_plot_title = (
        f"Initial Skewness: {round(initial_skew, round_to)}"
    )

    if abs(initial_skew) > skew_threshold and initial_skew > 0:
        skew_power_transformation = {}
        powers = np.linspace(1.01, max_power, num_power_iterations)

        for power in powers:
            skew = np.power(df_temp[feature], 1 / power).skew()
            skew_power_transformation[power] = skew

        power_min_skew = min(
            skew_power_transformation,
            key=lambda k: abs(skew_power_transformation[k]),
        )
        best_skew = skew_power_transformation[power_min_skew]

        feature_power_name = f"{round(1/power_min_skew, round_to)}"
        transformed[feature_power_name] = calculate_transformations(
            df_temp[[feature]], "power", 1 / power_min_skew
        )[0]
        results[feature_power_name] = round(best_skew, round_to)
        skewness_power_plot_title = "Skewness after raising to 1/"
        skewness_power_plot_title += f"{round(power_min_skew, 3)}: "
        skewness_power_plot_title += f"{round(best_skew, 4)}"

        feature_log_name = "log"
        min_value = df_temp[feature].min()
        if min_value < 0:
            correction_value = abs(min_value) + 1
            df_temp[[feature]] = df_temp[[feature]] + correction_value
        transformed[feature_log_name], log_skew = calculate_transformations(
            df_temp[[feature]], "log"
        )
        results[feature_log_name] = log_skew
        skewness_log_plot_title = (
            f"Skewness after log transformation: {round(log_skew, round_to)}"
        )

        feature_yeo_name = "yeo"
        transformed[feature_yeo_name], yeo_skew = calculate_transformations(
            df_temp[[feature]], "yeo"
        )
        results[feature_yeo_name] = yeo_skew
        skewness_yeo_plot_title = "Skewness after Yeo-Johnson transformation: "
        skewness_yeo_plot_title += f"{round(yeo_skew, round_to)}"

        if plot:
            plot_transformations(
                {
                    skewness_power_plot_title: transformed[feature_power_name],
                    skewness_log_plot_title: transformed[feature_log_name],
                    skewness_yeo_plot_title: transformed[feature_yeo_name],
                },
                df_temp[feature],
                initial_skew_plot_title,
            )

    elif abs(initial_skew) > skew_threshold and initial_skew < 0:
        feature_power2_name = str(2)
        transformed[feature_power2_name], skew2 = calculate_transformations(
            df_temp[[feature]], "power", 2
        )
        results[feature_power2_name] = skew2
        skewness_power2_plot_title = (
            f"Skewness after raising to 2: {round(skew2, round_to)}"
        )

        feature_power3_name = str(3)
        transformed[feature_power3_name], skew3 = calculate_transformations(
            df_temp[[feature]], "power", 3
        )
        results[feature_power3_name] = skew3
        skewness_power3_plot_title = (
            f"Skewness after raising to 3: {round(skew3, round_to)}"
        )

        feature_log_name = "log"
        correction_value = df_temp[feature].max() + 1
        shifted = correction_value - df_temp[feature]
        transformed[feature_log_name], log_skew = calculate_transformations(
            shifted, "log"
        )
        results[feature_log_name] = log_skew
        skewness_log_plot_title = (
            f"Skewness after log transformation: {round(log_skew, round_to)}"
        )

        feature_yeo_name = "yeo"
        transformed[feature_yeo_name], yeo_skew = calculate_transformations(
            df_temp[[feature]], "yeo"
        )
        results[feature_yeo_name] = yeo_skew
        skewness_yeo_plot_title = "Skewness after Yeo-Johnson transformation: "
        skewness_yeo_plot_title += f"{round(yeo_skew, round_to)}"

        if plot:
            plot_transformations(
                {
                    skewness_power2_plot_title: transformed[
                        feature_power2_name
                    ],
                    skewness_power3_plot_title: transformed[
                        feature_power3_name
                    ],
                    skewness_log_plot_title: transformed[feature_log_name],
                    skewness_yeo_plot_title: transformed[feature_yeo_name],
                },
                df_temp[feature],
                initial_skew_plot_title,
            )

    else:
        return {}, {}

    return transformed, results


def select_best_transformation(
    results: dict, final_threshold: float, verbose: bool = True
) -> tuple[str, bool]:
    """
    Select the transformation that produced the minimum skewness

    Parameters:
        results (dict): A dictionary containing the skewness values for
            different transformations.
        final_threshold (float): The threshold for considering a transformation
            successful.

    Returns:
        tuple: A tuple containing the best transformation and a boolean
            indicating success.
    """
    best_transformation = min(results, key=lambda k: abs(results[k]))
    best_skew = results[best_transformation]
    if verbose:
        logging.info(f"Best transformation: {best_transformation}")
        logging.info(f"Skew after transformation: {best_skew}")
    success = abs(best_skew) <= final_threshold
    if verbose:
        if success:
            logging.info("Transformation successful")
        else:
            logging.info("Transformation failed. Converting to binary")
    return best_transformation, success


def apply_final_transformation(
    df: pd.DataFrame,
    transformed: dict,
    best_tranformation: str,
    feature: str,
    subsampled: bool,
    skew_positive: bool,
    round_to: int = 4,
) -> pd.DataFrame:
    """
    Apply the best transformation to correct the skewness.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the feature to transform.
        transformed (dict): A dictionary containing the transformed feature
            series.
        best_tranformation (str): The best transformation to apply.
        feature (str): The name of the feature to transform.
        subsampled (bool): A boolean indicating whether the feature is
            subsampled.
        skew_positive (bool): A boolean indicating whether the feature is
            positively skewed.
        round_to (int): The number of decimal places to round the skew result.

    Returns:
        pd.DataFrame: The DataFrame with the transformed feature added.
    """

    # If subsampled is false, the transformed dictionary contains the
    # transformed series for each transformation
    # applied to the entire dataset. Otherwise, the transformations were
    # applied to the subsampled dataset

    if subsampled:

        if best_tranformation == "log":
            if skew_positive:
                if df[feature].min < 0:
                    df[[feature]] = df[[feature]] + abs(df[[feature]].min) + 1
                df[feature + "_transformed"] = calculate_transformations(
                    df[[feature]], "log"
                )[0]
            else:
                correction_value = df[feature].max() + 1
                shifted = correction_value - df[feature]
                df[feature + "_transformed"] = calculate_transformations(
                    shifted, "log"
                )[0]

        elif best_tranformation == "yeo":
            df[feature + "_transformed"] = calculate_transformations(
                df[[feature]], "yeo"
            )[0]

        else:
            power = round(float(best_tranformation), round_to)
            df[feature + "_transformed"] = calculate_transformations(
                df[[feature]], "power", power
            )[0]
    # subsampled is false, the transformation were already applied to the
    # entire dataset
    else:
        df[feature + "_transformed"] = transformed[best_tranformation]

    return df


def fallback_to_binary(df, feature, skew_positive=True):
    """
    Convert the feature to binary if it is positively skewed or negatively
    skewed

    Parameters:
        df (pd.DataFrame): The DataFrame containing the feature to transform.
        feature (str): The name of the feature to transform.
        skew_positive (bool): A boolean indicating whether the feature is
            positively skewed.

    Returns:
        pd.DataFrame: The DataFrame with the transformed feature added.
    """

    binary_feature = f"{feature}_binary"
    if skew_positive:
        df[binary_feature] = (df[feature] != df[feature].min()).astype(int)

    else:
        df[binary_feature] = (df[feature] == df[feature].max()).astype(int)

    return df


def correct_skew(
    df,
    feature,
    max_power=50,
    initial_skew_threshold=0.5,
    final_skew_threshold=0.1,
    plot_all_transformations=False,
    plot_transformed_feature=False,
    verbose=True,
):
    """
    Automatically detects skewness in a DataFrame feature, applies the best
        transformation
    to reduce skewness, or converts it to binary if transformation fails.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the feature to transform.
        feature (str): The name of the feature to transform.
        max_power (int): The maximum power to apply to the feature
            (only valid for positively skewed features).
        initial_skew_threshold (float): The threshold for applying
            transformations.
        final_skew_threshold (float): The threshold for considering a
            transformation successful.
        plot_all_transformations (bool): A boolean indicating whether to plot
            the results for all transformations.
        plot_transformed_feature (bool): A boolean indicating whether to plot
            the transformed feature.
        verbose (bool): A boolean indicating whether to print verbose output.

    Returns:
        pd.DataFrame: The DataFrame with the transformed feature added.
    """
    df, df_temp, subsampled = remove_nans_and_subsample(df, feature, verbose)
    skew = df[feature].skew()
    if verbose:
        logging.info(f"Initial Skewness: {round(skew, 4)}")

    if abs(skew) < initial_skew_threshold:
        if verbose:
            logging.info(
                "Feature not sufficiently skewed. No transformation applied."
            )
        return {}, {}

    transformed, results = apply_transformations(
        df_temp,
        feature,
        initial_skew=skew,
        max_power=max_power,
        plot=plot_all_transformations,
        skew_threshold=initial_skew_threshold,
        num_power_iterations=100,
    )

    best_transformation, success = select_best_transformation(
        results, final_skew_threshold
    )

    if not success:
        if verbose:
            message = "Could not reduce skewness below threshold. "
            message += f"Converting '{feature}' to binary."
            logging.info(message)
        df_out = fallback_to_binary(df, feature, skew_positive=(skew > 0))
    else:
        df_out = apply_final_transformation(
            df,
            transformed,
            best_transformation,
            feature,
            subsampled,
            skew_positive=(skew > 0),
        )

    if plot_transformed_feature:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(df_out[feature], kde=True, stat="density", ax=axs[0])
        axs[0].set_title(feature)
        if success:
            sns.histplot(
                df_out[feature + "_transformed"],
                kde=True,
                stat="density",
                ax=axs[1],
            )
            axs[1].set_title(feature + "_transformed")
        else:
            sns.countplot(data=df_out, x=feature + "_binary", ax=axs[1])
            axs[1].set_title(feature + "_binary")
        plt.show()

    return df_out, results


if __name__ == "__main__":
    
    airline = load_csv_from_data("airline/train.csv")
    airline_tranformed, resultados = correct_skew(
        airline,
        "Arrival Delay in Minutes",
        plot_all_transformations=True,
        plot_transformed_feature=True,
        verbose=True,
     )
    '''
    insurance = load_csv_from_data("insurance/insurance.csv")
    insurance_tranformed, resultados = correct_skew(
        insurance,
        "charges",
        plot_all_transformations=True,
        plot_transformed_feature=True,
        verbose=True,
    )
    '''
