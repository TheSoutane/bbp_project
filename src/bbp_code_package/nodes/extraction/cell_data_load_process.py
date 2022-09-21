from typing import Any, Dict

import numpy as np
import pandas as pd
from kedro.extras.datasets.pandas import CSVDataSet

import bbp_code_package.nodes.extraction.mat_file_extraction as mat_extraction


def load_preprocess_mat_file(cell_source_csv: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Loops over the files to load and aggregate them together

    |  :param cell_source_csv: csv file containing the list of cells to  load
    |  :param parameters: dict of pipeline parameters
    |  :return: nothing, dataframe is saved through the data catalog

    """
    cell_type_column = parameters["cellinfo"]["cell_type_column"]
    cell_id_column = parameters["cellinfo"]["cell_id_column"]
    specie_col = parameters["cellinfo"]["specie_col"]

    cell_list = collect_cells_to_load(cell_source_csv, parameters)
    mat_cell_list = get_mat_name(cell_list)

    concatenated_dataframe = pd.DataFrame()

    for mat_cell in mat_cell_list:
        df_aggreg = pd.DataFrame()
        (cellinfo, protocol_data_list) = mat_extraction.extract_preformat_mat_file(
            mat_cell, parameters
        )

        df_aggreg = pd.concat(protocol_data_list, axis=1).copy()
        df_aggreg.fillna(-1500, inplace=True)
        df_aggreg[cell_type_column] = cellinfo[cell_type_column]
        df_aggreg[cell_id_column] = cellinfo[cell_id_column]
        df_aggreg[specie_col] = cellinfo[specie_col]

        # concatenated_dataframe, df_aggreg = match_columns(concatenated_dataframe, df_aggreg)

        concatenated_dataframe = pd.concat(
            [concatenated_dataframe, df_aggreg], sort=False, ignore_index=True
        ).copy()

    if parameters["refresh_df"]:
        cells_extracted_raw = CSVDataSet(
            filepath="data/02_intermediate/cells_extracted_raw.csv"
        )
        cells_extracted_raw.save(concatenated_dataframe)
    return concatenated_dataframe


def match_columns(df_1: pd.DataFrame, df_2: pd.DataFrame):
    """
    |  Ensures that the columns of the 2 df to concat are matching
    |  :param df_1: 1st df to match
    |  :param df_2: 2nd df to match
    |  :return: matched dataframe
    """

    missing_to_1 = [t for t in df_2.columns if not (t in df_1.columns)]
    missing_to_2 = [t for t in df_1.columns if not (t in df_2.columns)]

    df_missing_to_1 = pd.DataFrame(
        np.nan, index=range(df_1.shape[0]), columns=missing_to_1
    )
    df_missing_to_2 = pd.DataFrame(
        np.nan, index=range(df_2.shape[0]), columns=missing_to_2
    )

    df_1 = pd.concat([df_1, df_missing_to_1], axis=1)
    df_2 = pd.concat([df_2, df_missing_to_2], axis=1)

    return df_1, df_2


def collect_cells_to_load(data: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Extract the list of cells to be loaded from the dedicated file
    |  :param data: data table with cell ID and quality flag
    |  :param parameters: dict of pipeline parameters
    |  :return: list of cells to load
    """
    id_column = parameters["cells_list"]["id_column"]
    status_column = parameters["cells_list"]["status_column"]
    cell_list = data.loc[data[status_column] == 1][id_column].tolist()

    return cell_list


def get_mat_name(cell_list: list):
    """
    |  Build name of .mat file to load from cell ID
    |  :param cell_list: List of cells to consider
    |  :return: updated list containing the file names instead of the cell ID
    """
    out_list = []
    for cell in cell_list:
        out_list.append(f"aCell{cell}.mat")
    return out_list
