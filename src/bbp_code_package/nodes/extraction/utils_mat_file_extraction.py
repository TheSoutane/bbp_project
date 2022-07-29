import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import io

####
# In this script, functions extracting data from the .mat files and
# aggregating it in one dataframe are encoded.

####


def idrest_features_extraction(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Extract and reformat the features from the idrest protocol
    :param df:data to process
    :param parameters:
    :return:
        Dataframe with cleaned and formatted protocol data
    """
    preprocessing_columns = protocol_dict["preprocessing_columns"]
    repetitions_column = protocol_dict["repetitions_column"]

    features_df = df.loc[df[repetitions_column] == 0].copy()

    out_features = protocol_dict["out_features_base"].copy()

    features_df = features_df[preprocessing_columns]
    features_df = flatten_columns(features_df, protocol_dict)

    features_df, out_features = compute_peak_frequency(
        features_df, protocol_dict, out_features
    )

    has_missing_peaks = get_missing_peaks(features_df, protocol_dict)

    features_df, out_features = compute_ap_potential_columns(
        features_df, protocol_dict, out_features
    )

    features_df, out_features = compute_isi_columns(
        features_df, protocol_dict, out_features
    )

    # import pdb; pdb.set_trace()
    features_df = pivot_protocol_table(features_df, protocol_dict, out_features)
    features_df["has_missing_peaks"] = has_missing_peaks
    return features_df


def apwaveform_features_extraction(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Extract and reformat the features from the APWaveform protocol
    :param df:data to process
    :param parameters:
    :return:
        Dataframe with cleaned and formatted protocol data
    """
    preprocessing_columns = protocol_dict["preprocessing_columns"]
    repetitions_column = protocol_dict["repetitions_column"]

    features_df = df.loc[df[repetitions_column] == 0].copy()

    out_features = protocol_dict["out_features_base"].copy()

    features_df = features_df[preprocessing_columns]
    features_df = flatten_columns(features_df, protocol_dict)

    features_df, out_features = compute_peak_frequency(
        features_df, protocol_dict, out_features
    )
    has_missing_peaks = get_missing_peaks(features_df, protocol_dict)

    features_df, out_features = compute_ap_potential_columns(
        features_df, protocol_dict, out_features
    )

    features_df, out_features = compute_isi_columns(
        features_df, protocol_dict, out_features
    )
    features_df, out_features = compute_ahp_columns(
        features_df, protocol_dict, out_features
    )
    # import pdb; pdb.set_trace()
    features_df = pivot_protocol_table(features_df, protocol_dict, out_features)
    features_df["has_missing_peaks"] = has_missing_peaks

    return features_df


def get_missing_peaks(features_df, protocol_dict):
    """

    :param features_df:
    :param protocol_dict:
    :return:
    """
    stim_treshold = protocol_dict["stim_treshold"]
    stim_col = protocol_dict["stim_col"]
    spike_col = protocol_dict["spike_col"]

    sample_df = features_df.loc[features_df[stim_col] > stim_treshold]

    return sample_df[spike_col].min() == 0


def flatten_df(df: pd.DataFrame):
    """
    Flatten all the columns of a datafram to facilitate data extraction

    :param df: dataframe to be flattened
    :return: flattened dataframe
    """

    df_flat = df.copy()
    df_flat = df_flat.apply(lambda x: x[0][0])

    return df_flat


def get_min_list(x, n=2):
    """
    Creates list of given size including the input data
    :param x: input value
    :param n: size of list to be created
    :return: list of size n including the values of x
    """
    if isinstance(x, int):
        return [x] + [0] * (n - 1)
    if isinstance(x, list):
        if len(x) < n:
            return x + [0] * (n - len(x))
        else:
            return x
    else:
        return x


def load_mat_file(filepath: str, filename: str):
    """

    :param filepath: path leading to the folder where .mat files are stored
    :param filename: name of the file to load

    :return:loaded .mat file
    """

    return io.loadmat(os.path.join(filepath, filename))


def extract_cell_info(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    extracts cellinfo from the imported .mat file
    :param df:
    :param parameters:
    :return: cleaned cellinfo data as pandas series
    """
    cell_category = parameters["cellinfo"]["category"]
    cell_sub_category = parameters["cellinfo"]["sub_category"]

    cellInfo = pd.DataFrame(df[cell_category][0][0][cell_sub_category].ravel())
    cellInfo_flat = flatten_df(cellInfo)

    return cellInfo_flat


def extract_protocol_data(
    df: pd.DataFrame,
    parameters: Dict[str, Any],
    protocol_dict: Dict[str, Any],
    protocols_index_dict: Dict[str, Any],
):
    """

    :param df: raw data
    :param parameters:
    :return: non formatted dataframe with data aggregated at protocol level
    """
    cell_category = parameters["cellinfo"]["category"]
    protocol_category = parameters["cellinfo"]["protocol"]
    repetitions_column = parameters["repetitions_column"]
    protocol_tag = protocol_dict["protocol_tag"]
    protocol_id = protocols_index_dict[protocol_tag]

    # import pdb; pdb.set_trace()
    protocol_name = parameters["protocol_name"]
    protocol_raw = df[cell_category][0][0][protocol_category][0][protocol_id]

    protocol_df = pd.DataFrame(protocol_raw[0][0][1][0])
    protocol_df[protocol_name] = protocol_raw[0][0][0][0]
    protocol_df = protocol_df.reset_index().rename(
        columns={"index": repetitions_column}
    )

    return protocol_df


def get_protocol_dict(raw_file: pd.DataFrame, parameters: Dict[str, Any]):
    """
    Get the index corresponding to each protocol and stores it in a dict
    :param raw_file: raw .mat file that was just loaded
    :return: Dict with the protocol name as entry and the index as object
    """

    cell_category = parameters["cellinfo"]["category"]
    protocol_category = parameters["cellinfo"]["protocol"]

    protocols_data = raw_file[cell_category][0][0][protocol_category][0]

    index_dict = {}
    for protocol_name, index_val in zip(protocols_data, range(protocols_data.shape[0])):
        index_dict[protocol_name[0][0][0][0]] = index_val

    return index_dict


def flatten_at_trace_level(trace_df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Reshape elements at trace level (size must match across all features)
    and then explodes the data to get a df at trace level
    :param df: dataframe_to_flatten
    :param protocol_dict: dict of parameters at protocol level
    :return: dataframe flattened at trace level
    """
    trace_dict = protocol_dict["trace_level"]

    stimulation_column = trace_dict["stimulation_column"]
    stimulations_to_keep = trace_dict["stimulations_to_keep"]
    trace_level_columns = trace_dict["trace_level_columns"]

    size_list = []
    for col in trace_level_columns:
        size_list.append(trace_df[col].apply(lambda x: x.shape[1]).max())
    req_size = max(size_list)

    for col in trace_level_columns:
        trace_df[col] = trace_df[col].apply(empty_array_corrector, args=(req_size,))

    # import pdb; pdb.set_trace()

    trace_df = trace_df.explode(trace_level_columns).explode(trace_level_columns)

    trace_df_filtered = trace_df.loc[
        trace_df[stimulation_column].isin(stimulations_to_keep)
    ]
    return trace_df_filtered


def empty_array_corrector(x: np.ndarray, req_size: int):
    """
    reshape arrays to the required size
    :param x: Array to reshape
    :param req_size: size of reshape
    :return: reshaped/padded array
    """
    if x.shape == np.zeros(shape=(0, 0)).shape:
        return np.zeros(shape=(1, req_size))
    if x.shape[1] < req_size:
        pad_size = req_size - x.shape[1]
        return np.pad(x, [(0, 0), (pad_size, 0)], mode="constant", constant_values=0)
    else:
        return x


def double_extract(x):
    """

    :param x: feature to extract
    :return: value with corrected feature
    """
    return x.item()


def single_extract(x):
    """

    :param x: feature to extract
    :return: value with corrected feature
    """
    return x.flatten().tolist()


def flatten_columns(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Flatten the object in columns. This allow an easier use of data  later
    :param df: data to format
    :param parameters:
    :return: dataframe where columns value is an integer or a list
    """
    double_extract_features = protocol_dict["double_extract_features"]
    single_extract_features = protocol_dict["single_extract_features"]
    for col in double_extract_features:
        df[col] = df[col].apply(double_extract)

    for col in single_extract_features:
        df[col] = df[col].apply(single_extract)

    return df


def compute_peak_frequency(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features
):
    """

    :param df: data from which frequency should be exctracted
    :param parameters:
    :return: data with new column, peak frequency
    """

    freq_dict = protocol_dict["freq_dict"]
    stim_lenght = freq_dict["stim_lenght"]
    stim_start = freq_dict["stim_start"]
    stim_end = freq_dict["stim_end"]
    peak_frequency = freq_dict["peak_frequency"]
    spikecount = freq_dict["spikecount"]

    df[stim_lenght] = (
        df[stim_end] - df[stim_start]
    ) / 1000  # /1000 to go from milisec to sec
    df[peak_frequency] = df[spikecount] / df[stim_lenght]

    df.loc[df[peak_frequency] == np.inf, peak_frequency] = 0
    out_features.extend([stim_lenght, peak_frequency])

    return df, out_features


def compute_ap_potential_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    :param df: data from which frequency should be exctracted
    :param parameters:
    :return: data with new column, peak frequency
    """

    ap_potential_dict = protocol_dict["ap_potential"]
    AP_begin_voltage = ap_potential_dict["AP_begin_voltage"]
    min_list_size = ap_potential_dict["min_list_size"]
    AP_mean_wo_first = ap_potential_dict["AP_mean_wo_first"]
    AP_stdev_wo_first = ap_potential_dict["AP_stdev_wo_first"]
    AP_mean = ap_potential_dict["AP_mean"]
    AP_stdev = ap_potential_dict["AP_stdev"]
    first_AP_voltage = ap_potential_dict["first_AP_voltage"]
    second_AP_voltage = ap_potential_dict["second_AP_voltage"]

    df[AP_begin_voltage] = df[AP_begin_voltage].apply(
        lambda x: get_min_list(x, min_list_size)
    )

    # features covering all AP
    df[AP_mean] = df[AP_begin_voltage].apply(lambda x: np.mean(x))
    df[AP_stdev] = df[AP_begin_voltage].apply(lambda x: np.std(x))

    # features of 1st and 2nd spikes
    df[first_AP_voltage] = df[AP_begin_voltage].apply(lambda x: x[0])
    df[second_AP_voltage] = df[AP_begin_voltage].apply(lambda x: x[1])

    # features excluding first spike
    df[AP_mean_wo_first] = df[AP_begin_voltage].apply(lambda x: np.mean(x))
    df[AP_stdev_wo_first] = df[AP_begin_voltage].apply(lambda x: np.std(x))

    out_features.extend(
        [
            first_AP_voltage,
            second_AP_voltage,
            AP_mean,
            AP_stdev,
            AP_mean_wo_first,
            AP_stdev_wo_first,
        ]
    )

    return df, out_features


def compute_isi_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    :param df: data from which frequency should be exctracted
    :param parameters:
    :return: data with new column, peak frequency
    """

    isi_dict = protocol_dict["ISI_values"]
    ISI_values = isi_dict["ISI_values"]
    first_ISI = isi_dict["first_ISI"]
    second_ISI = isi_dict["second_ISI"]
    min_list_size = isi_dict["min_list_size"]

    df[ISI_values] = df[ISI_values].apply(lambda x: get_min_list(x, int(min_list_size)))
    df[first_ISI] = df[ISI_values].apply(lambda x: x[0])
    df[second_ISI] = df[ISI_values].apply(lambda x: x[1])

    out_features.extend([ISI_values, first_ISI, second_ISI])
    return df, out_features


def compute_ahp_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    :param df: data from which frequency should be exctracted
    :param parameters:
    :return: data with new column, peak frequency
    """

    ahp_dict = protocol_dict["ahp_values"]
    tau_mean = ahp_dict["tau_mean"]
    tau_stdev = ahp_dict["tau_stdev"]
    ahp_fall_tau = ahp_dict["ahp_fall_tau"]
    first_fall_tau = ahp_dict["first_fall_tau"]
    second_fall_tau = ahp_dict["second_fall_tau"]

    # features covering all APtau
    df[tau_mean] = df[ahp_fall_tau].apply(lambda x: np.mean(x))
    df[tau_stdev] = df[ahp_fall_tau].apply(lambda x: np.std(x))

    df[ahp_fall_tau] = df[ahp_fall_tau].apply(lambda x: get_min_list(x))

    df[first_fall_tau] = df[ahp_fall_tau].apply(lambda x: x[0])
    df[second_fall_tau] = df[ahp_fall_tau].apply(lambda x: x[1])

    out_features.extend([tau_mean, tau_stdev, first_fall_tau, second_fall_tau])
    return df, out_features


def pivot_protocol_table(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    :param df:
    :param parameters:
    :param out_features:
    :return:
    """

    pivot_dict = protocol_dict["pivot_dict"]
    index_columns = pivot_dict["index_columns"]
    pivot_index = pivot_dict["pivot_index"]
    pivot_columns = pivot_dict["pivot_columns"]

    pivot_input = df[out_features + index_columns]

    rename_dict = dict(zip(out_features, [f"{col}_stim" for col in out_features]))
    pivot_input_final = pivot_input.rename(columns=rename_dict)

    features_df_flat = pivot_input_final.pivot_table(
        index=pivot_index,
        columns=pivot_columns,
        values=[f"{col}_stim" for col in out_features],
    )

    features_df_flat.columns = [
        f"{column[0]}_{column[1]}" for column in features_df_flat.columns
    ]
    features_df_flat.reset_index(inplace=True)

    return features_df_flat
