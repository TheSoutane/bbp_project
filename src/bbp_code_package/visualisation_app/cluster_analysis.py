import dash
import dash_core_components as dcc
from dash import html

import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output

LIGHT_BLUE = "#B5D1FF"
DARK_BLUE = "#4A90FF"
LIGHT_ORANGE = "#ffeab7"
DARK_ORANGE = "#ffc94a"
LIGHT_GREY = "#ededed"

df = pd.read_csv("data/05_clustering/clustering_output_unsc.csv")

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    dbc.themes.BOOTSTRAP,
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

features_dropdown = dcc.Dropdown(
    id="features_dropdown",
    multi=True,
    options=[{"label": s, "value": s} for s in df.columns],
    value=["apwaveform_AP_mean_stim_180", "species_mouse", "species_rat"],
)

clusters_dropdown = dcc.Dropdown(
    id="clusters_dropdown",
    multi=False,
    options=[{"label": s, "value": s} for s in df.columns if ("cluster" in s)],
    value="6_clusters",
)


def _get_overview_children():
    return [
        # html.H3("Segment Overview"),
        features_dropdown,
        clusters_dropdown,
        html.Table(id="adjustable-test-table"),
        # slider,
    ]


def _get_table(aggregate, cluster_col):
    header = [html.Th("")] + [html.Th(col) for col in aggregate.columns]

    body_items = []
    row_names = list(aggregate.index)
    for row_name in row_names:
        row_items = [html.Th(row_name)]

        mean = aggregate.loc[row_name, "mean_val"]
        row_items.append(
            html.Td(
                mean,
                style={"background-color": LIGHT_GREY},
            )
        )
        std = aggregate.loc[row_name, "stdev"]
        row_items.append(html.Td(std, style={"background-color": LIGHT_GREY}))

        cluster_count = cluster_col.split("_")[0]

        for cluster_col_num in range(int(cluster_count)):
            value = aggregate.loc[row_name, cluster_col_num]

            if value > mean + (2 * std):
                td = html.Td(value, style={"background-color": DARK_BLUE})
            elif value < mean - (2 * std):
                td = html.Td(value, style={"background-color": DARK_ORANGE})
            elif value > mean + (1 * std):
                td = html.Td(value, style={"background-color": LIGHT_BLUE})
            elif value < mean - (1 * std):
                td = html.Td(value, style={"background-color": LIGHT_ORANGE})
            else:
                td = html.Td(value)
            row_items.append(td)

        row = html.Tr(row_items)
        body_items.append(row)

    return [
        html.Table(
            # Header
            header
            +
            # Body
            body_items
        )
    ]


app.layout = html.Div(
    [
        html.H1(children="Cells Segmentation"),
        # dcc.Tabs(
        #     id="main-tabs",
        #     children=[_get_overview_tab(), _get_deepdive_tab(), _get_cuts_tab()],
        # ),
    ]
    + _get_overview_children()
)


@app.callback(
    [
        Output("adjustable-test-table", "children"),
    ],
    # [Input("num-clusters-slider", "value"), Input("feature-view-selection", "value")],
    [Input("features_dropdown", "value"), Input("clusters_dropdown", "value")],
)
def create_overview_table(features, cluster):
    df_features = df[features + [cluster]].copy()
    df_aggreg = df_features.groupby(cluster).mean()

    counts_df = (
        df_features[[cluster, features[0]]].groupby(cluster).count().reset_index()
    )
    counts_df.rename(columns={features[0]: "count"}, inplace=True)

    df_aggreg = df_aggreg.reset_index().merge(counts_df, on=cluster).transpose()

    stats_df = df_features.mean().reset_index()
    stats_df["stdev"] = df_features.std().values
    stats_df.columns = ["index", "mean_val", "stdev"]

    df_aggreg = df_aggreg.reset_index().merge(stats_df, on="index", how="left")

    df_aggreg = df_aggreg.round(2)

    first_column = df_aggreg.pop("stdev")
    df_aggreg.insert(0, "stdev", first_column)
    first_column = df_aggreg.pop("mean_val")
    df_aggreg.insert(0, "mean_val", first_column)

    # clusters_list = [t for t in df_aggreg.columns if isinstance(t, (int,float))]
    df_aggreg.set_index("index", inplace=True)
    table = _get_table(df_aggreg, cluster)

    return table
    # return [html.Table(
    #     # Header
    #     [html.Tr([html.Th(col) for col in df_aggreg.columns])] +
    #     # Body
    #     [html.Tr([
    #         html.Td(df_aggreg.iloc[i][col]) for col in df_aggreg.columns
    #     ]) for i in range(len(df_aggreg))]
    # )]


if __name__ == "__main__":
    app.run_server(debug=True)
