import pandas as pd
import plotly
import plotly.express as px
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import bbp_code_package.nodes.utils as utils


def filter_per_cell_type(df, celltype, aggregation_level):
    filtered_df = df.loc[df[aggregation_level] == celltype].copy()
    return filtered_df


def get_html_box_plot(
    df: pd.DataFrame, cols_to_analyse, colour_col: str, savepath: str, savefile: str
):
    """
    Plot scattermatrix HTML file over defined dict

    :param df: data to plot
    :param cols_to_analyse: columns to be analysed
    :param colour_col: col used to group data
    :param savepath: path to savefolder
    :param savefile: Path to save sub-file
    :return: Html Box plot of each of the cols_to_analyse, saved in the indicated folder
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
    :param df: Dataframe whose columns needs to be filtered
    :param features_markers: List of markers ('strings') to be used for extraction
    :return: List of columns containing at least one of the markers
    """
    cols_list = []
    for marker in features_markers:
        cols_list.extend([t for t in df.columns if (marker in t)])
    cols_list_unique = [*set(cols_list)]
    return cols_list_unique


def run_stat_test(df, stat_test_dict, test_type):
    """
    Performs t-test for each cat of cell.
    :param df: Data to test
    :param t_test_dict: parameters
    :return: Dataframe with t test for each selected feature
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
            celltype_array = df.loc[df[aggregation_level] == celltype][col]
            other_array = df.loc[df[aggregation_level] != celltype][col]

            if test_type == "t_test":
                stat, p = ttest_ind(
                    other_array, celltype_array, nan_policy="omit", equal_var=False
                )
                if not isinstance(p, float):
                    p = 0.5
            if test_type == "mann_whitney":
                if celltype_array.isna().sum() == len(celltype_array):
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
    :param df:
    :param t_test_dict:
    :return:
    """
    features_markers = t_test_dict["features_markers"]
    report_save_path = t_test_dict["report_save_path"]
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
    :param df:
    :param stat_dict:
    :return:
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
