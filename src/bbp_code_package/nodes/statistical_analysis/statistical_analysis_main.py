import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import bbp_code_package.nodes.statistical_analysis.statistical_analysis_utils as stat_utils


def run_statistical_analysis_1_v_1(df, parameters):

    stat_dict = parameters["statistical_analysis_dict"]
    mann_whitney_dict = stat_dict["mann_whitney_dict"]
    cells_to_drop = mann_whitney_dict['cells_to_drop']
    aggregation_level = mann_whitney_dict["aggregartion_level"]
    features_markers = mann_whitney_dict["features_markers"]

    celltypes_to_compare = [t for t in df[aggregation_level].unique() if (t not in cells_to_drop)]

    cols_to_analyse = stat_utils.get_cols_from_feat_markers(df, features_markers)
    n_celltype = len(celltypes_to_compare)
    feature_matrix_dict = {}
    df_concat_list= []
    for feature in cols_to_analyse:

        feature_array = np.zeros((n_celltype, n_celltype))
        for i in range(n_celltype):
            for j in range(i+1, n_celltype):
                cell_i = celltypes_to_compare[i]
                cell_j = celltypes_to_compare[j]
                cell_i_df = stat_utils.filter_per_cell_type(df, cell_i, aggregation_level)[feature]
                cell_j_df = stat_utils.filter_per_cell_type(df, cell_j, aggregation_level)[feature]

                if (cell_i_df.isna().sum() == cell_i_df.shape[0]) or (cell_j_df.isna().sum() == cell_j_df.shape[0]):
                    print('pass')
                    p = 0.5
                else:
                    stat, p = mannwhitneyu(
                        cell_i_df, cell_j_df, nan_policy="omit"
                    )
                feature_array[i][j] = p
                feature_array[j][i] = p
                df_concat_list.extend([[cell_i, cell_j, feature, p], [cell_j, cell_i, feature, p]])

        feature_matrix_dict[feature] = feature_array

        # other function
       # for feature in feature_matrix_dict.keys():
       #     f, ax = plt.subplots(figsize=(20, 20))
       #     ax = sns.heatmap(feature_matrix_dict[feature])
       #     ax.set_title(feature)
       #     ax.set_xlabel(celltypes_to_compare)
       #     ax.set_ylabel(celltypes_to_compare)
       #     filename = f"{feature}.pdf"
       #     plt.savefig(filename)

    stat_sign_df = pd.DataFrame(df_concat_list, columns=['cell_i', 'cell_j', 'feature', 'p_value'])
    stat_sign_df.sort_values(['cell_i', 'cell_j', 'p_value'], inplace=True)

    stat_synthesis_list = []
    for cell_i in stat_sign_df.cell_i.unique():
        for cell_j in stat_sign_df.cell_j.unique():
            temp_df = stat_sign_df.loc[(stat_sign_df.cell_i == cell_i) & (stat_sign_df.cell_j == cell_j)][['feature', 'p_value']].copy()
            stat_synthesis_list.append([cell_i, cell_j, temp_df.nsmallest(columns='p_value', n=3).values])

    stat_synthesis_df = pd.DataFrame(stat_synthesis_list, columns=['cell_i', 'cell_j', 'differenciators'])
    return stat_sign_df, stat_synthesis_df


def run_statistical_analysis_1_v_all(df, parameters):

    """
    Orchestrator running the statistical analysis on the formatted data
    :param df:
    :param parameters:
    :return:
    """
    stat_dict = parameters["statistical_analysis_dict"]
    features_markers = stat_dict["boxplot_dict"]["features_markers"]
    class_features = stat_dict["class_features"]
    savepath = stat_dict["savepath"]
    boxplot_savefile = stat_dict["boxplot_dict"]["boxplot_savefile"]


    cols_to_analyse = stat_utils.get_cols_from_feat_markers(df, features_markers)
    # stat_utils.get_html_box_plot(
    #    df, cols_to_analyse, class_features, savepath, boxplot_savefile
    # )

    run_t_test = stat_dict["run_t_test"]
    run_mann_whitney = stat_dict["run_mann_whitney"]

    stat_significance_df = pd.DataFrame()
    if run_t_test:
        t_test_dict = stat_dict["t_test_dict"]
        stat_significance_t_test = stat_utils.run_stat_test(df, t_test_dict, "t_test")
        stat_significance_t_test["test_type"] = "t_test"
        stat_utils.plot_stat_test(stat_significance_t_test, t_test_dict)
        stat_significance_df = pd.concat(
            [stat_significance_df, stat_significance_t_test]
        )

    if run_mann_whitney:
        mann_whitney_dict = stat_dict["mann_whitney_dict"]
        stat_significance_m_w = stat_utils.run_stat_test(
            df, mann_whitney_dict, "mann_whitney"
        )
        stat_significance_m_w["test_type"] = "mann_whitney"
        stat_utils.plot_stat_test(stat_significance_m_w, mann_whitney_dict)
        stat_significance_df = pd.concat([stat_significance_df, stat_significance_m_w])

    statistical_analysis = stat_utils.get_statistical_analysis(
        stat_significance_df, stat_dict
    )

    return stat_significance_df, statistical_analysis
