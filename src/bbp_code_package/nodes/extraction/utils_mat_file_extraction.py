import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import io
from itertools import product

####
# In this script, functions extracting data from the .mat files and
# aggregating it in one dataframe are encoded.

####


def features_extraction(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    |  Extract and reformat the features from the APWaveform protocol
    |  :param df: dataframe to be processeddata to process
    |  :param parameters: dict of pipeline parameters
    |  :return:
    |  Dataframe with cleaned and formatted protocol data
    """
    preprocessing_columns = get_preprocessing_columns(protocol_dict)
    repetitions_column = protocol_dict["repetitions_column"]
    features_df = df.loc[df[repetitions_column] == 0].copy()

    out_features = protocol_dict["out_features_base"].copy()
    features_df = features_df[preprocessing_columns]
    features_df = flatten_columns(features_df, protocol_dict)

    features_df, out_features = compute_peak_frequency(
        features_df, protocol_dict, out_features
    )
    has_missing_peaks = get_missing_peaks(features_df, protocol_dict)

    features_df, out_features = compute_peak_columns(
        features_df, protocol_dict, out_features
    )

    features_df = pivot_protocol_table(features_df, protocol_dict, out_features)
    features_df["has_missing_peaks"] = has_missing_peaks

    return features_df


def get_missing_peaks(df: pd.DataFrame, protocol_dict):
    """

    |  :param df: dataframe to be processed
    |  :param protocol_dict: protocol parameters sub-dict
    |  :return: features w peaks
    """
    stim_treshold = protocol_dict["stim_treshold"]
    stim_col = protocol_dict["stim_col"]
    spike_col = protocol_dict["spike_col"]

    sample_df = df.loc[df[stim_col] > stim_treshold]

    return sample_df[spike_col].min() == 0


def flatten_df(df: pd.DataFrame):
    """
    |  Flatten all the columns of a datafram to facilitate data extraction

    |  :param df: dataframe to be flattened
    |  :return:  flattened dataframe
    """

    df_flat = df.copy()
    df_flat = df_flat.apply(lambda x: x[0][0])

    return df_flat


def get_min_list(x, n=2):
    """
    |  Creates list of given size including the input data
    |  :param x: input value
    |  :param n: size of list to be created
    |  :return:  list of size n including the values of x
    """
    if (
        isinstance(x, int)
        | isinstance(x, np.int32)
        | isinstance(x, np.int64)
        | isinstance(x, float)
    ):
        output = [x] + [0] * (n - 1)
    elif isinstance(x, np.ndarray):
        try:
            x = [t for t in x[0]]
        except:
            x = [t for t in x]
        if len(x) < n:
            output = x + [0] * (n - len(x))
        else:
            output = x

    elif isinstance(x, list):
        if len(x) < n:
            output = x + [0] * (n - len(x))
        else:
            output = x
    else:
        output = x

    return output


def load_mat_file(filepath: str, filename: str):
    """

    |  :param filepath: path leading to the folder where .mat files are stored
    |  :param filename: name of the file to load

    |  :return: loaded .mat file
    """

    return io.loadmat(os.path.join(filepath, filename))


def extract_cell_info(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Extracts cellinfo from the imported .mat file
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:  cleaned cellinfo data as pandas series
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

    |  :param  df: raw data
    |  :param parameters: dict of pipeline parameters
    |  :return:  non formatted dataframe with data aggregated at protocol level
    """
    cell_category = parameters["cellinfo"]["category"]
    protocol_category = parameters["cellinfo"]["protocol"]
    repetitions_column = parameters["repetitions_column"]
    protocol_tag = protocol_dict["protocol_tag"]
    protocol_id = protocols_index_dict[protocol_tag]

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
    |  Get the index corresponding to each protocol and stores it in a dict
    |  :param raw_file: raw .mat file that was just loaded
    |  :return:  Dict with the protocol name as entry and the index as object
    """

    cell_category = parameters["cellinfo"]["category"]
    protocol_category = parameters["cellinfo"]["protocol"]

    protocols_data = raw_file[cell_category][0][0][protocol_category][0]

    index_dict = {}
    for protocol_name, index_val in zip(protocols_data, range(protocols_data.shape[0])):
        index_dict[protocol_name[0][0][0][0]] = index_val

    return index_dict


def get_spikecount(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    |  Count number of spikes in case the "spikecount" feature is missinggit
    |  :param df: dataframe to be processed
    |  :param protocol_dict:
    |  :return: df with created spikecount
    """
    recompute_spikecount = protocol_dict["recompute_spikecount"]
    if recompute_spikecount:
        recompute_col = protocol_dict["recompute_col"]
        spike_col = protocol_dict["spike_col"]
        df[spike_col] = df[recompute_col].apply(lambda x: count_spikes(x))

    return df


def extract_cheops(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Extracts the cheops protocol as it has an extra level of nesting (see experimental notebooks)
    |  :param df: dataframe to be processed (can be an array)
    |  :param protocol_dict:
    |  :return: cheops parameters dataframe
    """
    cheops_cols = protocol_dict['cheops_cols']

    df_temp = df.cheops[0][0]
    if df_temp.shape[0] == 0:
        cols = [f'{x}_{y}' for x, y in list(product(cheops_cols, [1,2,3]))]
        df_out = pd.DataFrame(columns=cols)

    else:
        df_out = pd.DataFrame(df_temp[cheops_cols]).copy()
        for col in cheops_cols:
            print(col)
            df_out[col] = df_out[col].apply(lambda x: cheops_extract(x)).copy()

        if protocol_dict['pivot']:
            df_out['pivot_index'] = 1
            df_out = df_out.reset_index().pivot(index = 'pivot_index', columns = 'index', values=cheops_cols).copy()
            df_out.columns = [f"{column[0]}_{column[1]}" for column in df_out.columns]
            df_out.reset_index(inplace=True)
            df_out.drop(columns=['pivot_index'], inplace=True)

    return df_out


def cheops_extract(x):
    if x.shape == (1,1):
        return x[0][0]
    if x.shape == (0,0):
        return np.nan
    else:
        return np.nan

def count_spikes(x):
    if isinstance(x, int):
        out = 1
    else:
        out = x.shape[1]
    return np.array([[[], out]])


def flatten_at_trace_level(trace_df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Reshape elements at trace level (size must match across all features)
    and then explodes the data to get a df at trace level
    |  :param df: dataframe to be processed dataframe_to_flatten
    |  :param protocol_dict: dict of parameters at protocol level
    |  :return:  dataframe flattened at trace level
    """
    trace_level_columns = get_trace_level_cols(protocol_dict)

    trace_dict = protocol_dict["trace_level"]

    stimulation_column = trace_dict["stimulation_column"]
    stimulations_to_keep = trace_dict["stimulations_to_keep"]
    fixed_value_columns = trace_dict["fixed_value_columns"]

    size_list = []
    for col in trace_level_columns:
        size_list.append(trace_df[col].apply(lambda x: x.shape[1]).max())
    req_size = max(size_list)

    for col in trace_level_columns:
        trace_df[col] = trace_df[col].apply(empty_array_corrector, args=(req_size,))

    for col in fixed_value_columns:
        trace_df[col] = trace_df[col].apply(
            extract_single_val_for_explode, args=(req_size,)
        )

    trace_level_columns = trace_level_columns + fixed_value_columns
    trace_df = trace_df.explode(trace_level_columns).explode(trace_level_columns)
    trace_df_filtered = trace_df.loc[
        trace_df[stimulation_column].isin(stimulations_to_keep)
    ]

    return trace_df_filtered


def extract_single_val_for_explode(x, list_dim: list):

    try:
        while not isinstance(x.item(), (int, float)):
            x = x.item()

        output = [[x.item()] * list_dim]
    except:
        output = [[np.nan] * list_dim]
    return output


def empty_array_corrector(x: np.ndarray, req_size: int):
    """
    reshape arrays to the required size
    |  :param x: Array to reshape
    |  :param req_size: size of reshape
    |  :return:  reshaped/padded array
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

    |  :param x: feature to extract
    |  :return:  value with corrected feature
    """
    return x.item()


def single_extract(x):
    """

    |  :param x: feature to extract
    |  :return:  value with corrected feature
    """
    return x.flatten().tolist()


def flatten_columns(df: pd.DataFrame, protocol_dict: Dict[str, Any]):
    """
    Flatten the object in columns. This allow an easier use of data  later
    |  :param df: dataframe to be processed data to format
    |  :param parameters: dict of pipeline parameters
    |  :return:  dataframe where columns value is an integer or a list
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

    |  :param df: dataframe to be processed data from which frequency should be exctracted
    |  :param parameters: dict of pipeline parameters
    |  :return:  data with new column, peak frequency
    """
    if protocol_dict["compute_freq"]:
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


def compute_peak_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """
    Compute the peak frequency
    |  :param df: data from which frequency should be exctracted
    |  :param parameters: dict of pipeline parameters
    |  :return:  data with new column, peak frequency

    """

    peak_values_dict = protocol_dict["peak_values"]

    for feat in peak_values_dict.keys():
        feat_dict = peak_values_dict[feat]
        df, out_dict_feat = extract_first_sec_mean(df, feat_dict)
        out_features.extend(out_dict_feat)

    return df, out_features


def compute_other_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    |  :param df: data from which frequency should be exctracted
    |  :param parameters: dict of pipeline parameters
    |  :return:  data with new column, peak frequency

    """

    ap_potential_dict = protocol_dict["other_values"]

    for feat in ap_potential_dict.keys():
        feat_dict = ap_potential_dict[feat]
        df, out_dict_feat = extract_first_sec_mean(df, feat_dict)
        out_features.extend(out_dict_feat)

    return df, out_features


def compute_isi_columns(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """

    |  :param df: data from which frequency should be exctracted
    |  :param parameters: dict of pipeline parameters
    |  :return:  data with new column, peak frequency
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

    |  :param df: data from which frequency should be exctracted
    |  :param parameters: dict of pipeline parameters
    |  :return:  data with new column, peak frequency
    """

    ahp_dict = protocol_dict["ahp_values"]
    AHP_fall_tau_dict = ahp_dict["AHP_fall_tau"]
    min_AHP_voltage_dict = ahp_dict["min_AHP_voltage"]
    AHP_duration_dict = ahp_dict["AHP_duration"]

    df, AHP_fall_tau_feat = extract_first_sec_mean(df, AHP_fall_tau_dict)
    df, min_AHP_voltage_feat = extract_first_sec_mean(df, min_AHP_voltage_dict)
    df, AHP_duration_feat = extract_first_sec_mean(df, AHP_duration_dict)

    out_features.extend(AHP_fall_tau_feat + min_AHP_voltage_feat + AHP_duration_feat)

    return df, out_features


def extract_first_sec_mean(df: pd.DataFrame, feature_dict: Dict[str, Any]):
    """
    create the spikes features by selecting the data of the 1st, 2nd and average value
    |  :param df: dataframe to be processed
    |  :param feature_dict:
    |  :return: dataframe with new (spikes) features and list of columns names
    """

    feature_base_name = feature_dict["feature_base_name"]
    feature_nick_name = feature_dict["feature_nick_name"]

    df[f"mean_{feature_nick_name}"] = df[feature_base_name].apply(lambda x: np.mean(x))

    df[feature_base_name] = df[feature_base_name].apply(lambda x: get_min_list(x))

    df[f"first_{feature_nick_name}"] = df[feature_base_name].apply(lambda x: x[0])
    df[f"second_{feature_nick_name}"] = df[feature_base_name].apply(lambda x: x[1])

    return df, [
        f"mean_{feature_nick_name}",
        f"first_{feature_nick_name}",
        f"second_{feature_nick_name}",
    ]


def pivot_protocol_table(
    df: pd.DataFrame, protocol_dict: Dict[str, Any], out_features: list
):
    """
    |  Orchestrate table pivot
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :param out_features:
    |  :return: pivoted table
    """
    pivot_dict = protocol_dict["pivot_dict"]

    pivot_columns = pivot_dict["pivot_columns"]

    index_columns, pivot_index = get_column_and_pivot_index(protocol_dict)

    pivot_input = df[out_features + index_columns]

    rename_dict = dict(zip(out_features, [f"{col}_stim" for col in out_features]))
    pivot_input_final = pivot_input.rename(columns=rename_dict)
    pivot_values = [f"{col}_stim" for col in out_features]

    pivot_input_final.fillna(-1500, inplace=True)

    features_df_flat = pivot_input_final.pivot(
        index=pivot_index,
        columns=pivot_columns,
        values=pivot_values,
    )

    features_df_flat.columns = [
        f"{column[0]}_{column[1]}" for column in features_df_flat.columns
    ]
    features_df_flat.reset_index(inplace=True)
    return features_df_flat


def get_column_and_pivot_index(protocol_dict: Dict[str, Any]):
    """
    |  Get the 2 lists of columns required for the pivot fiunction
    |  :param protocol_dict: protocol parameters subdict
    |  :return: list of features for the pivot index and column parameters
    """
    preprocessing_features = protocol_dict["preprocessing_features"]
    pivot_columns = protocol_dict["pivot_dict"]["pivot_columns"]
    fixed_value_columns = protocol_dict["trace_level"]["fixed_value_columns"]

    stim_start = protocol_dict["freq_dict"]["stim_start"]
    stim_end = protocol_dict["freq_dict"]["stim_end"]
    spikecount = protocol_dict["freq_dict"]["spikecount"]

    cols_to_neglect = [stim_start, stim_end, spikecount]

    index_columns = fixed_value_columns + preprocessing_features
    index_columns = [t for t in index_columns if not (t in cols_to_neglect)]

    pivot_index = [t for t in index_columns if not (t in pivot_columns)]

    return np.unique(index_columns).tolist(), np.unique(pivot_index).tolist()


def get_preprocessing_columns(protocol_dict: Dict[str, Any]):
    """
    |  Collect the list of columns that are required for the preprocessing different subgroups
    |  :param protocol_dict: protocol parameters subdict
    |  :return: list of columns for preprocessing
    """
    preprocessing_features = protocol_dict["preprocessing_features"]
    double_extract_features = protocol_dict["double_extract_features"]
    single_extract_features = protocol_dict["single_extract_features"]

    fixed_value_columns = protocol_dict["trace_level"]["fixed_value_columns"]

    peak_values = list(protocol_dict["peak_values"].keys())

    return np.unique(
        preprocessing_features
        + double_extract_features
        + single_extract_features
        + fixed_value_columns
        + peak_values
    ).tolist()


def get_trace_level_cols(protocol_dict: Dict[str, Any]):
    """
    |  Collect the list of columns that are at trace level from the different subgroups
    |  :param protocol_dict:
    |  :return: List of trace level columns
    """
    preprocessing_features = protocol_dict["preprocessing_features"]
    double_extract_features = protocol_dict["double_extract_features"]
    single_extract_features = protocol_dict["single_extract_features"]

    peak_values = list(protocol_dict["peak_values"].keys())

    return np.unique(
        preprocessing_features
        + double_extract_features
        + single_extract_features
        + peak_values
    ).tolist()
