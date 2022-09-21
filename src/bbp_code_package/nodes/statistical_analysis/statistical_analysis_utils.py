import pandas as pd
import plotly
import plotly.express as px
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import bbp_code_package.nodes.utils as utils
import numpy as np
from collections import Counter


def get_stat_synthesis(stat_sign_df: pd.DataFrame, n_smallest, drop_stdev=True):
    """
    Extracts instigts fom the raw statistical analysis
    |  :param stat_sign_df:
    |  :return:
    """
    if drop_stdev == True:
        stat_sign_df["is_stdev"] = stat_sign_df["feature"].apply(lambda x: "stdev" in x)
        stat_sign_df = stat_sign_df.loc[~stat_sign_df["is_stdev"]]

    stat_synthesis_list = []
    for cell_i in stat_sign_df.cell_i.unique():
        for cell_j in stat_sign_df.cell_j.unique():
            temp_df = stat_sign_df.loc[
                (stat_sign_df.cell_i == cell_i) & (stat_sign_df.cell_j == cell_j)
            ][["feature", "p_value"]].copy()
            stat_synthesis_list.append(
                [
                    cell_i,
                    cell_j,
                    temp_df.nsmallest(columns="p_value", n=n_smallest).values,
                ]
            )

    stat_synthesis_df = pd.DataFrame(
        stat_synthesis_list, columns=["cell_i", "cell_j", "differenciators"]
    )

    stat_synthesis_df["diff_simplified"] = stat_synthesis_df["differenciators"].apply(
        lambda x: [t[0].split("_stim")[0] for t in x]
    )

    stat_synthesis_df["diff_simplified"] = stat_synthesis_df["diff_simplified"].apply(
        lambda x: Counter(x).most_common()
    )

    for rank in range(n_smallest):
        stat_synthesis_df[f"diff_{rank}"] = stat_synthesis_df["differenciators"].apply(
            lambda x: safely_get_value(x, rank)
        )

    stat_synthesis_df.drop(columns=["differenciators"], inplace=True)
    return stat_synthesis_df


def safely_get_value(x, i):
    """
    extract values from the nsmallest features
    |  :param x:
    |  :param i:
    |  :return:
    """
    if 0 == x.shape[0]:
        print("alternative")
        return np.NAN
    else:
        return x[i]


def run_1_V_1_stat_analysis(
    df, celltypes_to_compare, cols_to_analyse, aggregation_level
):
    """
    Runs the 1V1 test for each combination of cell types/feature combination.
    |  :param df: dataframe to be processed
    |  :param celltypes_to_compare:
    |  :param cols_to_analyse:
    |  :param aggregation_level:
    |  :return:
    """
    n_celltype = len(celltypes_to_compare)
    feature_matrix_dict = {}
    df_concat_list = []

    for feature in cols_to_analyse:

        feature_array = np.zeros((n_celltype, n_celltype))
        for i in range(n_celltype):
            for j in range(i + 1, n_celltype):
                cell_i = celltypes_to_compare[i]
                cell_j = celltypes_to_compare[j]
                cell_i_df = filter_per_cell_type(df, cell_i, aggregation_level)[feature]
                cell_j_df = filter_per_cell_type(df, cell_j, aggregation_level)[feature]

                if (cell_i_df.isna().sum() == cell_i_df.shape[0]) or (
                    cell_j_df.isna().sum() == cell_j_df.shape[0]
                ):
                    print("pass")
                    p = 0.5
                else:
                    stat, p = mannwhitneyu(cell_i_df, cell_j_df, nan_policy="omit")
                feature_array[i][j] = p
                feature_array[j][i] = p
                df_concat_list.extend(
                    [[cell_i, cell_j, feature, p], [cell_j, cell_i, feature, p]]
                )

        feature_matrix_dict[feature] = feature_array
    stat_sign_df = pd.DataFrame(
        df_concat_list, columns=["cell_i", "cell_j", "feature", "p_value"]
    )
    stat_sign_df.sort_values(["cell_i", "cell_j", "p_value"], inplace=True)

    return stat_sign_df


def filter_per_cell_type(df: pd.DataFrame, celltype, aggregation_level):
    """

    |  :param df: dataframe to be processed
    |  :param celltype:
    |  :param aggregation_level:
    |  :return:
    """
    filtered_df = df.loc[df[aggregation_level] == celltype].copy()
    return filtered_df


def get_html_box_plot(
    df: pd.DataFrame, cols_to_analyse, colour_col: str, savepath: str, savefile: str
):
    """
    Plot scattermatrix HTML file over defined dict

    |  :param df: data to plot
    |  :param cols_to_analyse: columns to be analysed
    |  :param colour_col: col used to group data
    |  :param savepath: path to savefolder
    |  :param savefile: Path to save sub-file
    |  :return:  Html Box plot of each of the cols_to_analyse, saved in the indicated folder
    """
    for column in cols_to_analyse:
        print(column)
        filename = f"{savefile}\{column}.html"
        plotly.offline.plot(
            px.box(
                df,
                x=colour_col,
                y=column,
                color=colour_col,
                title=column,
                hover_name="id",
                points="all",
            ),
            filename=f"{savepath}\{filename}",
            auto_open=False,
            config={"scrollZoom": True},
        )


def get_cols_from_feat_markers(df, features_markers):
    """
    List columns with a specific marker
    |  :param df: Dataframe whose columns needs to be filtered
    |  :param features_markers: List of markers ('strings') to be used for extraction
    |  :return:  List of columns containing at least one of the markers
    """
    cols_list = []
    for marker in features_markers:
        cols_list.extend([t for t in df.columns if (marker in t)])
    cols_list_unique = [*set(cols_list)]
    return cols_list_unique


def run_stat_test(df, stat_test_dict, test_type):
    """
    Performs t-test for each cat of cell.
    |  :param df: Data to test
    |  :param t_test_dict: parameters
    |  :return:  Dataframe with t test for each selected feature
    """

    features_markers = stat_test_dict["features_markers"]
    aggregation_level = stat_test_dict["aggregartion_level"]

    cells_to_drop = stat_test_dict["cells_to_drop"]
    cellType_col = stat_test_dict["cellType_col"]

    df = df.loc[~df[cellType_col].isin(cells_to_drop)]

    cols_to_test = get_cols_from_feat_markers(df, features_markers)

    df_list = []
    for col in cols_to_test:
        dict_test = {}
        for celltype in df.cellType.unique():
            celltype_array = df.loc[df[aggregation_level] == celltype][col].copy()
            other_array = df.loc[df[aggregation_level] != celltype][col].copy()

            if test_type == "t_test":
                stat, p = ttest_ind(
                    other_array, celltype_array, nan_policy="omit", equal_var=False
                )
                if not isinstance(p, float):
                    p = 0.5
            if test_type == "mann_whitney":
                if celltype_array.isna().sum() == len(celltype_array):
                    p = 0.5
                elif len(celltype_array.unique()) < 3:
                    p = 0.5
                else:

                    stat, p = mannwhitneyu(
                        other_array, celltype_array, nan_policy="omit"
                    )

            dict_test[celltype] = p
        temp_df = pd.DataFrame.from_dict(dict_test, orient="index")
        temp_df.columns = [col]
        df_list.append(temp_df)
    stat_significance_df = pd.concat(df_list, axis=1)
    stat_significance_df[cellType_col] = [t for t in dict_test.keys()]
    return stat_significance_df


def plot_stat_test(df, t_test_dict):
    """
    plot the heatmap of the statistical tests
    |  :param df: dataframe to be processed
    |  :param t_test_dict:
    |  :return:
    """
    features_markers = t_test_dict["features_markers"]
    report_save_path = t_test_dict["report_save_path"]
    report_save_path = utils.get_path(report_save_path)

    graph_folder = t_test_dict["graph_folder"]
    t_test_report_name = t_test_dict["t_test_report_name"]

    files_list = []
    for marker in features_markers:
        sub_group = [t for t in df.columns if (marker in t)]
        sub_group.sort()
        f, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(df[sub_group])
        filename = f"{report_save_path}\{graph_folder}\{marker}.pdf"
        plt.savefig(filename)
        files_list.append(filename)

    utils.merge_pdfs(
        files_list, writing_path=f"{report_save_path}\{t_test_report_name}.pdf"
    )


def get_statistical_analysis(df, stat_dict):
    """
    synthetize the results of the statistical tests
    |  :param df: dataframe to be processed
    |  :param stat_dict:
    |  :return:
    """
    non_feat_cols = stat_dict["non_feat_cols"]
    p_min = stat_dict["p_min"]
    p_max = stat_dict["p_max"]
    filler_value = stat_dict["filler_value"]
    class_features = stat_dict["class_features"]

    concat_list = []
    for test in df.test_type.unique():
        df_test = df.loc[df.test_type == test].copy()

        for feature in [t for t in df_test.columns if (t not in non_feat_cols)]:
            df_feature = df_test[[feature, class_features]].copy()
            df_feature = df_feature.loc[~(df_feature[feature] == filler_value)].copy()

            if df_feature.shape[0] == 0:
                pass
            else:
                non_sign_list = df_feature.loc[df_feature[feature] < p_min][
                    class_features
                ].tolist()
                sign_list = df_feature.loc[df_feature[feature] > p_max][
                    class_features
                ].tolist()
                non_sign_count = sum(df_feature[feature] < p_min)
                sign_count = sum(df_feature[feature] > p_max)
                non_sign_share = non_sign_count / df_feature.shape[0]
                sign_share = sign_count / df_feature.shape[0]
                n_largest = df_feature.nlargest(columns=feature, n=3).values
                n_smallest = df_feature.nsmallest(columns=feature, n=3).values

                feat_df = pd.DataFrame(
                    {
                        "non_sign_count": [non_sign_count],
                        "sign_count": [sign_count],
                        "non_sign_share": [non_sign_share],
                        "sign_share": [sign_share],
                        "n_largest": [n_largest],
                        "n_smallest": [n_smallest],
                        "feature": [feature],
                        "test": [test],
                        "non_sign_list": [non_sign_list],
                        "sign_list": [sign_list],
                    }
                )
                concat_list.append(feat_df)

    statistical_report = pd.concat(concat_list)
    statistical_report_pivoted = statistical_report.pivot(
        index="feature", columns="test"
    )
    statistical_report_pivoted.columns = [
        f"{a}_r{b}" for a, b in statistical_report_pivoted.columns
    ]

    return statistical_report_pivoted
