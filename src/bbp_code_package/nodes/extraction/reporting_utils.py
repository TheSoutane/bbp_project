import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px
import bbp_code_package.nodes.utils as utils


def get_report_02_intermediate(df: pd.DataFrame, parameters: Dict[str, Any]):
    report_parameters = parameters["reports"]["report_02_intermediate"]
    outliers_df = get_numerical_data_report(df, report_parameters)

    return outliers_df


def get_report_03_primary(df: pd.DataFrame, parameters: Dict[str, Any]):
    # report_parameters = parameters["reports"]["report_03_primary"]
    # outliers_df = get_numerical_data_report(df, report_parameters)

    get_raw_data_anaysis(df, parameters)
    outliers_df = pd.DataFrame()
    return outliers_df


def get_raw_data_anaysis(df, parameters):

    # same stimulation
    stim_groups_dict = parameters["analysis"]["stim_groups_dict"]
    group_col_name_type = parameters["columns"]["group_col_name_type"]
    group_col_name_in_pc = parameters["columns"]["group_col_name_in_pc"]

    savefile_type = parameters["analysis"]["savefile_type"]
    savefile_in_pc = parameters["analysis"]["savefile_in_pc"]
    savepath = parameters["analysis"]["savepath"]

    get_html_scattermatrix_from_dict(
        df, stim_groups_dict, group_col_name_type, savepath, savefile_type
    )
    get_html_scattermatrix_from_dict(
        df, stim_groups_dict, group_col_name_in_pc, savepath, savefile_in_pc
    )


def get_html_scattermatrix_from_dict(
    df: pd.DataFrame, groups_dict, colour_col: str, savepath: str, savefile: str
):
    """
    Plot scattermatrix HTML file over defined dict
    :param df:
    :param groups_dict:
    :param colour_col:
    :return:
    """
    for protocol in groups_dict.keys():
        stim_numbers = groups_dict[protocol]
        protocol_cols = [x for x in df.columns if (protocol in x)]

        for number in stim_numbers:
            filename = f"{savefile}\{savefile}_{protocol}_{number}.html"

            cols = [x for x in protocol_cols if (str(number) in x)]
            plotly.offline.plot(
                px.scatter_matrix(df, cols, color=colour_col, title=filename),
                filename=f"{savepath}\{filename}",
                auto_open=False,
                config={"scrollZoom": True},
            )


def get_numerical_columns(df: pd.DataFrame):
    """
    returns numerical columns of a dataframe
    :param df: dataframe to analyse
    :return: list of numerical columns
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    return df.select_dtypes(include=numerics).columns.tolist()


def get_numerical_data_report(df, report_parameters: Dict[str, Any]):

    numerical_columns_list = get_numerical_columns(df)
    file_list = []

    file_writing_path = report_parameters["file_writing_path"]
    report_writing_path = report_parameters["report_writing_path"]
    report_dict = {}
    outliers_df = pd.DataFrame()

    for col in numerical_columns_list:
        median, treshold_max, treshold_min, out_of_tresh_ids = get_features_statistics(
            df, col, report_parameters
        )

        report_dict[col] = out_of_tresh_ids
        filename = os.path.join(file_writing_path, f"figure_{col}.pdf")
        file_list.append(filename)

        get_feature_plot(df, col, median, treshold_min, treshold_max, filename)
        outliers_df_temp = get_outliers_df(df, col, median, treshold_min, treshold_max)

        outliers_df = pd.concat([outliers_df, outliers_df_temp])

    utils.merge_pdfs(file_list, report_writing_path)

    return outliers_df


def get_features_statistics(
    df: pd.DataFrame, col: str, report_parameters: Dict[str, Any]
):
    """

    :param df:
    :param col:
    :return:
    """

    treshold_mult = report_parameters["treshold_mult"]

    median = df[col].median()
    stdev = df[col].std()
    treshold_max = median + stdev * treshold_mult
    treshold_min = median - stdev * treshold_mult

    out_of_tresh_ids = df.loc[df[col] > treshold_max]["id"].tolist()
    out_of_tresh_ids += df.loc[df[col] < treshold_min]["id"].tolist()
    return median, treshold_max, treshold_min, out_of_tresh_ids


def get_feature_plot(
    df: pd.DataFrame,
    col: str,
    median: float,
    treshold_min: float,
    treshold_max: float,
    filename: str,
):

    plt.figure()
    plt.hist(df[col], 30)
    plt.title(col)
    plt.axvline(x=median, linewidth=5, label="median", color="red")
    plt.axvline(
        x=treshold_max,
        linestyle="dotted",
        linewidth=3,
        label="treshold_max",
        color="red",
    )
    plt.axvline(
        x=treshold_min,
        linestyle="dotted",
        linewidth=3,
        label="treshold_min",
        color="red",
    )
    plt.savefig(filename)
    plt.clf


def get_outliers_df(
    df: pd.DataFrame,
    col: str,
    median: float,
    treshold_min: float,
    treshold_max: float,
):

    df_outliers = df.loc[(df[col] < treshold_min) | (df[col] > treshold_max)][
        [col, "id"]
    ].copy()
    df_outliers["feature"] = col
    df_outliers["median"] = median
    df_outliers["treshold_min"] = treshold_min
    df_outliers["treshold_max"] = treshold_max
    df_outliers.rename(columns={col: "feature_value"}, inplace=True)

    return df_outliers
