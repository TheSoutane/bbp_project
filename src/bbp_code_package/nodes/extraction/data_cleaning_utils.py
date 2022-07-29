from typing import Any, Dict

import pandas as pd


def clean_data_from_cleaning_dict(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    :param df:
    :param parameters:
    :return:
    """
    cleaning_dict = parameters["cleaning_dict"]

    for col in cleaning_dict.keys:
        col_dict = cleaning_dict[col]
        method = col_dict["method"]

        if method == "remove":
            df = df.loc[~(df[col] > col_dict["max_cap"])].copy()

        elif method == "other":
            df = df.loc[~(df[col] > col_dict["max_cap"])].copy()

        else:
            print(f"method for col {col} not properly implemented")
