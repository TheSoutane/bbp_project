import pandas as pd
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import numpy as np


def extract_clustering_features(df, parameters):

    cols_to_cluster = get_features_from_markers(df, parameters)
    return df[cols_to_cluster], cols_to_cluster


def get_features_from_markers(df, parameters):

    clustering_markers = parameters["clustering_dict"]["data_prep"][
        "clustering_markers"
    ]
    cols_to_cluster = []
    for marker in clustering_markers:
        temp_list = [t for t in df.columns if (marker in t)]
        cols_to_cluster.extend(temp_list)

    return cols_to_cluster


def scale_data(df, cols_to_cluster, parameters):
    df_cluster = df[cols_to_cluster].copy()
    df_scaled = df_cluster.copy()
    df_scaled = df_scaled.fillna(df_scaled.mean())

    scaler = StandardScaler().fit(df_scaled.values)
    arr_scaled = scaler.transform(df_scaled.values)

    df_scaled = pd.DataFrame(arr_scaled, columns=cols_to_cluster)

    return df_scaled, scaler


def get_normalization_per_group_dict(cols_to_cluster, parameters):

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


def get_normalization_per_group(df, normalization_per_group_dict):

    for key in normalization_per_group_dict.keys():
        for sub_key in normalization_per_group_dict[key].keys():
            feat_list = normalization_per_group_dict[key][sub_key]
            df[feat_list] = df[feat_list] / len(feat_list)

    return df


def get_cluster_analysis_output(df, cuttree_df):
    cluster_output_list = []
    df_cluster_all = pd.concat([df, cuttree_df], axis=1)
    for cluster_column in cuttree_df.columns:
        cutree = cuttree_df[cluster_column]
        df_cluster = pd.concat([df, cutree], axis=1)
        df_cluster.columns = df.columns.tolist() + ["cluster"]

        #scaler = StandardScaler()
        #df_cluster_norm = scaler.fit_transform(df_cluster)
        #df_cluster_norm = pd.DataFrame(df_cluster_norm, columns=df_cluster.columns)
        centroids = df_cluster.groupby("cluster").mean()
        centroids['cluster_column'] = cluster_column
        cluster_output_list.append(centroids)

        #centroids_norm = df_cluster_norm.groupby("cluster").mean()
    return df_cluster_all, pd.concat(cluster_output_list).transpose()


def run_agglomerative_clustering(df_scaled, n_clusters_list, fig_savepath, agg_method):

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
        dend = shc.dendrogram(agglomerative_output)

        plt.ylabel("intra-cluster noise")
        plt.savefig(f"{fig_savepath}/dendogram_{n_clusters}_clusters.pdf")
        plt.clf()

    plot_elbow_aggl_clustering(agglomerative_output, fig_savepath)

    return pd.concat(cutree_list, axis=1)


def plot_elbow_aggl_clustering(agglomerative_output, fig_savepath):
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