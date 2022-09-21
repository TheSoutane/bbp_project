import pandas as pd
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from joblib import dump, load
from typing import Any, Dict

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import numpy as np


def one_hot_encoding_input(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  One Hot encode descriptive columns for cluster analysis
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """
    cols_to_ohe = parameters["clustering_dict"]["data_prep"]["cols_to_ohe"]
    df_encoded = pd.get_dummies(data=df, columns=cols_to_ohe)
    return df_encoded


def extract_clustering_features(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  TBC: THIS FUNCTION MIGHT BE DEPRECATED
    |  Extract the columns used for the clustering distance based on tags
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """
    cols_to_cluster = get_features_from_markers(df, parameters)
    return df[cols_to_cluster], cols_to_cluster


def get_features_from_markers(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Collect all feature with a specific tag in it
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """
    clustering_markers = parameters["clustering_dict"]["data_prep"][
        "clustering_markers"
    ]
    cols_to_cluster = []
    for marker in clustering_markers:
        temp_list = [t for t in df.columns if (marker in t)]
        cols_to_cluster.extend(temp_list)

    return cols_to_cluster


def scale_data(df: pd.DataFrame, cols_to_cluster, parameters: Dict[str, Any]):
    """
    |  Scale input data using a standartscaler (more adapted for clustering)
    |  :param df: dataframe to be processed
    |  :param cols_to_cluster:
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """
    df_cluster = df[cols_to_cluster].copy()
    df_scaled = df_cluster.copy()
    df_scaled = df_scaled.fillna(df_scaled.mean())

    scaler = StandardScaler().fit(df_scaled.values)
    arr_scaled = scaler.transform(df_scaled.values)

    df_scaled = pd.DataFrame(arr_scaled, columns=cols_to_cluster)

    return df_scaled, scaler


def save_scaler(scaler, parameters: Dict[str, Any]):
    scaler_path = parameters["clustering_dict"]["scaler_path"]
    dump(scaler, scaler_path, compress=True)


def load_scaler(parameters: Dict[str, Any]):
    scaler_path = parameters["clustering_dict"]["scaler_path"]

    return load(scaler_path)


def unscale_df(scaled_df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Descale DF for the cluster analysis
    |  :param scaled_df:
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """
    scaler = load_scaler(parameters)
    unscaled_df = scaler.inverse_transform(scaled_df)

    return unscaled_df


def get_normalization_per_group_dict(cols_to_cluster: list, parameters: Dict[str, Any]):
    """
    |  Aggregate features in a dict for distance normalisation
    |
    |  Why ?
    |  We normalize "clustering groups" in order to balance clustering.
    |  Dummy example: We want the APWaveform AP_ style features (3 different feats) to have the same importance as the
    |  vHold. Therefore, The NORMALIZED AP features are divided by 3 so the total importance of the AP feats equal the
    |  one of vHold
    |  :param cols_to_cluster:
    |  :param parameters: dict of pipeline parameters
    |  :return: dict with features types as entry and features list as output. This will be used to scale
    |  the clustering distance
    """

    protocol_markers = parameters["clustering_dict"]["data_prep"]["protocol_markers"]
    clustering_markers = parameters["clustering_dict"]["data_prep"][
        "clustering_markers"
    ]
    normalization_per_group_dict = {}

    for col in cols_to_cluster:
        for p_marker in protocol_markers:
            if not (p_marker in normalization_per_group_dict.keys()):
                normalization_per_group_dict[p_marker] = {}
            if p_marker in col:
                break
        for marker in clustering_markers:
            if not (marker in normalization_per_group_dict[p_marker].keys()):
                normalization_per_group_dict[p_marker][marker] = []
            if marker in col:
                break
        normalization_per_group_dict[p_marker][marker] += [col]

    return normalization_per_group_dict


def get_normalization_per_group(
    df: pd.DataFrame, normalization_per_group_dict: Dict[str, Any]
):
    """
    |  Runs distance normalisation. Why ?
    |  We normalize "clustering groups" in order to balance clustering.
    |  Dummy example: We want the APWaveform AP_ style features (3 different feats) to have the same importance as the
    |  vHold. Therefore, The NORMALIZED AP features are divided by 3 so the total importance of the AP feats equal the
    |  one of vHold

    |  :param df: dataframe to be processed
    |  :param normalization_per_group_dict:
    |  :return: df with scaled columns (see descr for explanation)
    """

    for key in normalization_per_group_dict.keys():
        for sub_key in normalization_per_group_dict[key].keys():
            feat_list = normalization_per_group_dict[key][sub_key]
            df[feat_list] = df[feat_list] / len(feat_list)

    return df


def get_cluster_analysis_output(
    df: pd.DataFrame, df_unscaled: pd.DataFrame, cuttree_df: pd.DataFrame
):
    """
    |  Process the outputs of the clustering algorithm to produce the files used in the analysis
    |  :param df: dataframe to be processed
    |  :param df_unscaled:
    |  :param cuttree_df: Clustering output. It contains the clustering tree obtained after the use of
    |  agglomerative clustering
    |  :return:
    |  df_cluster_all: Df with all clustering outputs (multiple number of clusters)
    |  df_cluster_unsc_all: de-scaled df_cluster_all
    |  pd.concat(cluster_output_list).transpose(): clustering output
    """

    cluster_output_list = []

    df_cluster_all = pd.concat([df, cuttree_df], axis=1)
    df_cluster_unsc_all = pd.concat([df_unscaled, cuttree_df], axis=1)

    for cluster_column in cuttree_df.columns:
        cutree = cuttree_df[cluster_column]
        df_cluster = pd.concat([df, cutree], axis=1)
        df_cluster.columns = df.columns.tolist() + ["cluster"]

        centroids = df_cluster.groupby("cluster").mean()
        centroids["cluster_column"] = cluster_column
        cluster_output_list.append(centroids)

        # centroids_norm = df_cluster_norm.groupby("cluster").mean()
    return (
        df_cluster_all,
        df_cluster_unsc_all,
        pd.concat(cluster_output_list).transpose(),
    )


def run_agglomerative_clustering(
    df_scaled: pd.DataFrame, n_clusters_list: list, fig_savepath: str, agg_method: str
):
    """
   |   Orchestrates the processing of agglomerative clustering
    |  :param df_scaled:
    |  :param n_clusters_list: number of clusters to be analysed
    |  :param fig_savepath: Where to save the figures
    |  :param agg_method: Agglomeration method
    |  :return: List of cluster attributions of input dataframe
    """

    agglomerative_output = shc.linkage(df_scaled, method=agg_method)
    cutree_list = []
    for n_clusters in n_clusters_list:

        cutree_list.append(
            pd.DataFrame(
                shc.cut_tree(agglomerative_output, n_clusters=n_clusters),
                columns=[f"{n_clusters}_clusters"],
            )
        )
        plt.figure(figsize=(10, 7))
        plt.title("Customer Dendograms")
        plt.xlabel("cells")
        # dend = shc.dendrogram(agglomerative_output)

        plt.ylabel("intra-cluster noise")
        plt.savefig(f"{fig_savepath}/dendogram_{n_clusters}_clusters.pdf")
        plt.clf()

    plot_elbow_aggl_clustering(agglomerative_output, fig_savepath)

    return pd.concat(cutree_list, axis=1)


def plot_elbow_aggl_clustering(agglomerative_output: list, fig_savepath: str):
    """
    |  Plot the intra noise cluster to see the elbow plot and define the number of clusters to select during the analysis
    |  :param agglomerative_output:
    |  :param fig_savepath:
    |  :return: None, save elbow plot in the savepath
    """
    intr_cluster_noise_inv = agglomerative_output[-20:, 2]
    intr_cluster_noise = intr_cluster_noise_inv[::-1]
    idxs = np.arange(1, len(intr_cluster_noise) + 1)
    plt.plot(idxs, intr_cluster_noise)

    acceleration = np.diff(intr_cluster_noise, 2)  # 2nd derivative of the distances
    plt.plot(idxs[:-2] + 1, acceleration)
    plt.title("intra-cluster noise VS # of clusters")
    plt.xlabel("# of clusters")
    plt.ylabel("intra-cluster noise")
    plt.savefig(f"{fig_savepath}/intra_cluster_noise.pdf")
