import bbp_code_package.nodes.clustering.clustering_utils as clustering_utils
import pandas as pd
import bbp_code_package.nodes.utils as utils
from typing import Any, Dict


def prepare_clustering_input(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  TO BE IMPLEMENTED
    |  ADD DESCRIPTIVE FEATURES REQUIRED FOR CLUSTER ANALYSIS

    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters

    |  Returns: Primary dataframe with descriptive features (ex one hot encoding)

    """
    df_encoded = clustering_utils.one_hot_encoding_input(df, parameters)

    return df_encoded


def normalize_clustering_df(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Orchestrate the preparation of the clustering input

    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return: Normalized df
    """

    df, cols_to_cluster = clustering_utils.extract_clustering_features(df, parameters)

    df_scaled, scaler = clustering_utils.scale_data(df, cols_to_cluster, parameters)

    normalization_per_group_dict = clustering_utils.get_normalization_per_group_dict(
        cols_to_cluster, parameters
    )

    clustering_input = clustering_utils.get_normalization_per_group(
        df_scaled, normalization_per_group_dict
    )

    clustering_utils.save_scaler(scaler, parameters)

    return clustering_input


def run_clustering(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Orchestrate the data clustering
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    |  df_cluster: Input df (scaled) with cluster columns (can be used for analysis)
    |  df_cluster_unsc: de-scaled df_cluster
    |  centroids: centroids of the clusters
    """
    clustering_param = parameters["clustering_dict"]["clustering_param"]
    agg_method = clustering_param["agg_method"]
    n_clusters_list = clustering_param["n_clusters_list"]
    fig_savepath = clustering_param["fig_savepath"]
    fig_savepath = utils.get_path(fig_savepath)

    clustering_method = clustering_param["clustering_method"]

    df_out = df.copy()
    clustering_input = normalize_clustering_df(df, parameters)

    if clustering_method == "agglomerative":
        cuttree_df = clustering_utils.run_agglomerative_clustering(
            clustering_input, n_clusters_list, fig_savepath, agg_method
        )

    (
        df_cluster,
        df_cluster_unsc,
        centroids,
    ) = clustering_utils.get_cluster_analysis_output(
        clustering_input, df_out, cuttree_df
    )

    return df_cluster, df_cluster_unsc, centroids
