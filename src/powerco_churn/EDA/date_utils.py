"""
This module contains functions to parse and format dates
"""

import logging
from datetime import datetime

import pandas as pd


from powerco_churn.utils.logger_utils import configure_logging

configure_logging(log_file_name="date_utils.log")
logger = logging.getLogger(__name__)


def parse_and_format_dates(
    date_string: str,
    standard_format: str = "%Y-%m-%d",
    return_type: str = "string",
):
    """
    Convert a date string into a datetime object or a formatted string.

    The function tests if the input string matches any of the following
    date formats:

        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d %b %Y",
        "%B %d, %Y"

    If the input string doesn't follow any of these formats, the function will
    return None.
    It can be returned a datetime object or a string.

    Args:
        date_string: date in string format
        standard_format: format of data that will be returned
        return_type: Type of object to return, either 'datetime' or 'string'.

    Returns:
        Union[datetime, str, None]: Parsed date as a datetime object or a
        formatted string. Returns None if parsing fails.

    """

    # accepted
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y"]

    # @if the input is not a string, return None
    if not isinstance(date_string, str):
        return None

    if return_type not in ("string", "datetime"):
        print(
            f"{return_type} is not a valid return type. "
            "The formats accepted are string "
            "and datetime. String will be used as default"
        )
        return_type = "string"

    for expected_format in formats:
        try:
            # convert to datetime using the format specified in
            # expected_format
            parsed_date = datetime.strptime(date_string, expected_format)
            return (
                parsed_date
                if return_type == "datetime"
                else parsed_date.strftime(standard_format)
            )
        except ValueError:
            continue

    return None


def create_new_date_columns(
    df,
    features_list: list,
    calculate_difference: bool = True,
    reference_date: str = None,
):
    """
    Create new columns with extracted date parts and (optionally)
    date differences.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_list (list): List of column names containing dates.
        reference_date (str, optional): Date used to calculate difference.
            If None, the current date is used. Supports formats
            like "%Y-%m-%d", "%d-%m-%Y", etc.
        calculate_difference (bool): Whether to compute date differences.
          Default is True.

    Returns:
        pd.DataFrame: The original DataFrame with new date-related columns.
    """

    for feature in features_list:
        if feature not in df.columns:
            logging.warning("Feature '%s' not in DataFrame", feature)
            continue

        # Vectorized conversion: invalid parses become NaT
        converted = pd.to_datetime(df[feature], errors="coerce")

        # If nothing parsed, skip
        if converted.isna().all():
            logging.warning(
                "No valid dates in feature '%s'; skipping", feature
            )
            continue

        # Otherwise, replace the column and extract parts
        df[feature] = converted
        df[f"{feature}_year"] = converted.dt.year
        df[f"{feature}_month"] = converted.dt.month
        df[f"{feature}_day"] = converted.dt.day
        df[f"{feature}_weekday"] = converted.dt.day_name()

        if calculate_difference:
            if reference_date is None:
                parsed_reference_date = datetime.today()
            else:
                parsed_reference_date = None
                for fmt in [
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%m/%d/%Y",
                    "%d %b %Y",
                    "%B %d, %Y",
                ]:
                    try:
                        parsed_reference_date = datetime.strptime(
                            reference_date, fmt
                        )
                        break
                    except ValueError:
                        continue
                if parsed_reference_date is None:
                    parsed_reference_date = datetime.today()
                    logging.warning(
                        "Unrecognized reference_date format. Using current "
                        "date."
                    )

            df[f"actual_date - {feature}"] = (
                parsed_reference_date - df[feature]
            )
            df[f"actual_date - {feature} in days"] = (
                parsed_reference_date - df[feature]
            ).dt.days

    return df


if __name__ == "__main__":
    airbnb = load_csv_from_data("airbnb/listings.csv")
    airbnb = create_new_date_columns(airbnb, ["last_review"])
    print(airbnb.head())
