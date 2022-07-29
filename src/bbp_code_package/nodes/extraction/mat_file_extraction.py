from typing import Any, Dict

import pandas as pd

import bbp_code_package.nodes.extraction.utils_mat_file_extraction as mat_utils


def extract_preformat_mat_file(filename: str, parameters: Dict[str, Any]):
    """
    Exteract and preformat features from the .mat file
    :param filename: File to process
    :param parameters:
    :return:
        cellinfo: pandas series containing the high level cell information
        apwaveform_features: features extracted from the APWaveform protocol
    """

    print("_____________", filename)
    filepath = parameters["mat_file_path"]

    raw_file = mat_utils.load_mat_file(filepath, filename)

    cellinfo = mat_utils.extract_cell_info(raw_file, parameters)

    protocols_index_dict = mat_utils.get_protocol_dict(raw_file, parameters)
    if "APWaveform" in protocols_index_dict.keys():
        apwaveform_features = apwaveform_extraction(
            raw_file, parameters, protocols_index_dict
        )
        apwaveform_features["apwaveform_availability"] = 1
    else:
        apwaveform_features = pd.DataFrame()
        apwaveform_features["apwaveform_availability"] = 0

    if "IDRest" in protocols_index_dict.keys():
        idrest_features = idrest_extraction(raw_file, parameters, protocols_index_dict)
        idrest_features["idrest_availability"] = 1
    else:
        idrest_features = pd.DataFrame()
        idrest_features["idrest_availability"] = 0

    return cellinfo, apwaveform_features, idrest_features


def apwaveform_extraction(
    df: pd.DataFrame, parameters: Dict[str, Any], protocols_index_dict: Dict[str, Any]
):
    """
    Orchestrates the extraction and the formatting of the apwaveform data
    :param df: raw mat file
    :param parameters:
    :return: dataframe with ap_waveform features
    """
    protocol_dict = parameters["apwaveform_dict"]

    protocol_raw = mat_utils.extract_protocol_data(
        df, parameters, protocol_dict, protocols_index_dict
    )
    protocol_trace_level = mat_utils.flatten_at_trace_level(
        protocol_raw, protocol_dict=protocol_dict
    )
    apwaveform_features = mat_utils.apwaveform_features_extraction(
        protocol_trace_level, protocol_dict
    )
    return apwaveform_features.add_prefix("apwaveform_")


def idrest_extraction(
    df: pd.DataFrame, parameters: Dict[str, Any], protocols_index_dict: Dict[str, Any]
):
    """
    Orchestrates the extraction and the formatting of the idrest data
    :param df: raw mat file
    :param parameters:
    :return: dataframe with ap_waveform features
    """

    protocol_dict = parameters["idrest_dict"]

    protocol_raw = mat_utils.extract_protocol_data(
        df, parameters, protocol_dict, protocols_index_dict
    )
    protocol_trace_level = mat_utils.flatten_at_trace_level(
        protocol_raw, protocol_dict=protocol_dict
    )
    idrest_features = mat_utils.idrest_features_extraction(
        protocol_trace_level, protocol_dict
    )

    return idrest_features.add_prefix("idrest_")
