from typing import Any, Dict
import bbp_code_package.nodes.utils as utils
import pandas as pd

import bbp_code_package.nodes.extraction.utils_mat_file_extraction as mat_utils


def extract_preformat_mat_file(filename: str, parameters: Dict[str, Any]):
    """
    |  Extract and preformat features from the .mat file
    |  :param filename: File to process
    |  :param parameters: dict of pipeline parameters
    |  :return:
    |  cellinfo: pandas series containing the high level cell information
    |  apwaveform_features: features extracted from the APWaveform protocol
    """

    print("_____________", filename)
    mat_file_path = parameters["mat_file_path"]
    mat_file_path = utils.get_path(mat_file_path)

    raw_file = mat_utils.load_mat_file(mat_file_path, filename)

    cellinfo = mat_utils.extract_cell_info(raw_file, parameters)

    protocols_index_dict = mat_utils.get_protocol_dict(raw_file, parameters)

    features_extraction_dict = parameters["features_extraction_dict"]

    protocol_data_list = []
    for feature in features_extraction_dict.keys():
        protocol_tag = features_extraction_dict[feature]["protocol_tag"]
        protocol_marker = features_extraction_dict[feature]["protocol_marker"]
        print(protocol_tag)
        if protocol_tag in protocols_index_dict.keys():
            print("processing")
            protocol_features = features_extraction_orchestrator(
                raw_file, parameters, protocols_index_dict, protocol_marker
            )
            protocol_features[f"{protocol_marker}availability"] = 1
            if protocol_tag == "other":
                import pdb

                pdb.set_trace()
        else:
            print("missing")
            protocol_features = pd.DataFrame()
            protocol_features[f"{protocol_marker}availability"] = 0
        protocol_data_list.append(protocol_features)

    return cellinfo, protocol_data_list


def features_extraction_orchestrator(
    df: pd.DataFrame,
    parameters: Dict[str, Any],
    protocols_index_dict: Dict[str, Any],
    protocol_marker: str,
):
    """
    |  Orchestrates the extraction and the formatting of the apwaveform data
    |  :param df: dataframe to be processed raw mat file
    |  :param parameters: dict of pipeline parameters
    |  :return: dataframe with ap_waveform features
    """

    protocol_dict = parameters["features_extraction_dict"][f"{protocol_marker}dict"]
    protocol_raw = mat_utils.extract_protocol_data(
        df, parameters, protocol_dict, protocols_index_dict
    )

    if "cheops" in protocol_marker:

        df_features = mat_utils.extract_cheops(protocol_raw, protocol_dict)

    else:
        protocol_raw = mat_utils.get_spikecount(protocol_raw, protocol_dict)

        protocol_trace_level = mat_utils.flatten_at_trace_level(
            protocol_raw, protocol_dict=protocol_dict
        )

        df_features = mat_utils.features_extraction(protocol_trace_level, protocol_dict)

    return df_features.add_prefix(protocol_marker)
