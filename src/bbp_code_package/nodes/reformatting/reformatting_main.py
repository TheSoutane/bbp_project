from typing import Any, Dict

import pandas as pd

import bbp_code_package.nodes.reformatting.reform_utils as reform_utils


def reformat_dataframe(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Orchestratoe running the reformating functions
    |  :param df: Data to be reformatted
    |  :param parameters: dict of pipeline parameters
    |  :return:  Formatted dataframe
    """
    treshold_tau = parameters["reformatting_params"]["treshold_tau"]
    features_w_no_peaks = parameters["reformatting_params"]["features_w_no_peaks"]
    # group_mapping_params = parameters["reformatting_params"]["group_mapping_params"]
    df = reform_utils.tau_cap(df, treshold_tau)
    df = reform_utils.remove_features_w_no_peaks(df, features_w_no_peaks)
    df = reform_utils.remove_rows_w_missing_protocols(df)
    df = reform_utils.get_cell_group(df, parameters)
    df.fillna(0, inplace=True)

    return df
