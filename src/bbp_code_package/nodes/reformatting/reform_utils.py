import pandas as pd


def tau_cap(df: pd.DataFrame, treshold: int):
    """
    Limit the value of the tau columns to a specific treschold
    :param df: Data to reformat
    :param treshold: Max value of Tau column
    :return: Reformated DataFrame
    """
    tau_cols = [t for t in df.columns if ("tau" in t)]
    df[tau_cols] = df[tau_cols].clip(upper=treshold)
    # import pdb; pdb.set_trace()
    return df


def remove_features_w_no_peaks(df: pd.DataFrame, features_w_no_peaks: list):
    """
    Remove peak features at stimulations with limited peaks
    :param df: datafram to correct
    :param features_w_no_peaks: list of features to be removed
    :return: df
    """

    df.drop(columns=features_w_no_peaks, inplace=True)

    return df


def remove_rows_w_missing_protocols(df: pd.DataFrame):
    """
    Remove row (or cells) where one protocol is missing
    :param df:datafram to correct

    """
    protocols_availability_columns = [t for t in df.columns if ("availability" in t)]

    df["has_protocols"] = df[protocols_availability_columns].prod(axis=1)
    df_filtered = df.loc[df["has_protocols"] == 1].copy()
    df_filtered.drop(columns=["has_protocols"], inplace=True)
    return df


def get_cell_group(df: pd.DataFrame, parameters):
    """
    Map cell type to cell group
    :param df: datafram to correct
    :param group_mapping_params: Mapping dict
    :return: Df with new column of cell grou
    """
    group_mapping_dict_type = parameters["reformatting_params"]["group_mapping_params"][
        "group_mapping_dict_type"
    ]
    group_mapping_dict_in_pc = parameters["reformatting_params"][
        "group_mapping_params"
    ]["group_mapping_dict_in_pc"]

    group_col_name_in_pc = parameters["columns"]["group_col_name_in_pc"]
    group_col_name_type = parameters["columns"]["group_col_name_type"]
    cell_type_column = parameters["columns"]["cell_type_column"]

    df[group_col_name_type] = df[cell_type_column].map(group_mapping_dict_type)
    df[group_col_name_in_pc] = df[cell_type_column].map(group_mapping_dict_in_pc)

    return df
