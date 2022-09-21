import pandas as pd
from typing import Any, Dict

import bbp_code_package.nodes.statistical_analysis.statistical_analysis_utils as stat_utils
import bbp_code_package.nodes.utils as utils


def run_statistical_analysis_1_v_1(df: pd.DataFrame, parameters: Dict[str, Any]):
    """
    |  Orchestrate the 1V1 statistical test across the data
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    """

    stat_dict = parameters["statistical_analysis_dict"]
    mann_whitney_dict = stat_dict["mann_whitney_dict"]
    cells_to_drop = mann_whitney_dict["cells_to_drop"]
    aggregation_level = mann_whitney_dict["aggregartion_level"]
    features_markers = mann_whitney_dict["features_markers"]
    n_smallest = stat_dict["n_smallest"]
    celltypes_to_compare = [
        t for t in df[aggregation_level].unique() if (t not in cells_to_drop)
    ]

    cols_to_analyse = stat_utils.get_cols_from_feat_markers(df, features_markers)

    stat_sign_df = stat_utils.run_1_V_1_stat_analysis(
        df, celltypes_to_compare, cols_to_analyse, aggregation_level
    )
    # other function
    # for feature in feature_matrix_dict.keys():
    #     f, ax = plt.subplots(figsize=(20, 20))
    #     ax = sns.heatmap(feature_matrix_dict[feature])
    #     ax.set_title(feature)
    #     ax.set_xlabel(celltypes_to_compare)
    #     ax.set_ylabel(celltypes_to_compare)
    #     filename = f"{feature}.pdf"
    #     plt.savefig(filename)

    print("synthesis")
    stat_synthesis_df = stat_utils.get_stat_synthesis(stat_sign_df, n_smallest)

    return stat_sign_df, stat_synthesis_df


def run_statistical_analysis_1_v_all(df: pd.DataFrame, parameters: Dict[str, Any]):

    """
    |  Orchestrator running the statistical analysis on the formatted data
    |  :param df: dataframe to be processed
    |  :param parameters: dict of pipeline parameters
    |  :return:
    |  stat_significance_df: Dataframe containing the statistical significance of each celltype_1/celltype_2/feature
    |  statistical_analysis: synthesis of the statistical analysis
    """
    stat_dict = parameters["statistical_analysis_dict"]
    features_markers = stat_dict["boxplot_dict"]["features_markers"]
    class_features = stat_dict["class_features"]
    savepath = stat_dict["savepath"]
    savepath = utils.get_path(savepath)

    boxplot_savefile = stat_dict["boxplot_dict"]["boxplot_savefile"]

    cols_to_analyse = stat_utils.get_cols_from_feat_markers(df, features_markers)
    stat_utils.get_html_box_plot(
        df, cols_to_analyse, class_features, savepath, boxplot_savefile
    )

    run_t_test = stat_dict["run_t_test"]
    run_mann_whitney = stat_dict["run_mann_whitney"]

    stat_significance_df = pd.DataFrame()
    if run_t_test:
        t_test_dict = stat_dict["t_test_dict"]
        stat_significance_t_test = stat_utils.run_stat_test(df, t_test_dict, "t_test")
        stat_significance_t_test["test_type"] = "t_test"
        # stat_utils.plot_stat_test(stat_significance_t_test, t_test_dict)
        stat_significance_df = pd.concat(
            [stat_significance_df, stat_significance_t_test]
        )

    if run_mann_whitney:
        mann_whitney_dict = stat_dict["mann_whitney_dict"]
        stat_significance_m_w = stat_utils.run_stat_test(
            df, mann_whitney_dict, "mann_whitney"
        )
        stat_significance_m_w["test_type"] = "mann_whitney"
        # stat_utils.plot_stat_test(stat_significance_m_w, mann_whitney_dict)
        stat_significance_df = pd.concat([stat_significance_df, stat_significance_m_w])

    statistical_analysis = stat_utils.get_statistical_analysis(
        stat_significance_df, stat_dict
    )

    return stat_significance_df, statistical_analysis
