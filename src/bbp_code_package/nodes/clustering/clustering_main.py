import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt

import bbp_code_package.nodes.clustering.clustering_utils as clustering_utils


def prepare_clustering_input(df, parameters):

    df, cols_to_cluster = clustering_utils.extract_clustering_features(df, parameters)

    df_scaled, scaler = clustering_utils.scale_data(df, cols_to_cluster, parameters)

    normalization_per_group_dict = clustering_utils.get_normalization_per_group_dict(
        cols_to_cluster, parameters
    )

    clustering_input = clustering_utils.get_normalization_per_group(
        df_scaled, normalization_per_group_dict
    )

    return clustering_input


def run_clustering(df_scaled, parameters):

    clustering_param = parameters["clustering_dict"]["clustering_param"]
    agg_method = clustering_param["agg_method"]
    n_clusters_list = clustering_param["n_clusters_list"]
    fig_savepath = clustering_param["fig_savepath"]
    clustering_method = clustering_param["clustering_method"]

    if clustering_method == "agglomerative":
        cuttree_df = clustering_utils.run_agglomerative_clustering(
            df_scaled, n_clusters_list, fig_savepath, agg_method
        )

    (
        df_cluster,
        centroids,
    ) = clustering_utils.get_cluster_analysis_output(df_scaled, cuttree_df)

    return df_cluster,  centroids
