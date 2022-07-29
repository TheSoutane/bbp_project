import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Data from U.S. Congress, Joint Economic Committee, Social Capital Project.
df = pd.read_csv("data/03_primary/cells_reformated.csv")

app = dash.Dash(__name__)

# ------------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.ConfirmDialog(
            id="confirm-dialog",
            displayed=False,
            message="Please choose Dropdown variables!",
        ),
        html.H1(
            "Box or Scatter plot of neuron features", style={"textAlign": "center"}
        ),
        dcc.Dropdown(
            id="box_vs_scatter",
            options=["box plot", "scatterplot"],
            value="box plot",
        ),
        dcc.Dropdown(
            id="color_dropdown",
            options=["species", "cell_group_type", "cell_group_in_pc", "cellType"],
            value="cellType",
        ),
        dcc.Dropdown(
            id="x_feature",
            options=[{"label": s, "value": s} for s in df.columns],
            value="apwaveform_AP_mean_stim_180",
        ),
        dcc.Dropdown(
            id="y_feature",
            options=[{"label": s, "value": s} for s in df.columns],
            value="apwaveform_AP_mean_stim_180",
        ),
        dcc.Checklist(
            options=["FS", "DA_type", "IN", "PC", "Others", "Amygdala"],
            value=["FS", "IN", "PC", "Others", "Amygdala"],
            id="filter",
        ),
        dcc.Graph(id="my-chart", figure={}),
    ]
)


# ------------------------------------------------------------------------
@app.callback(
    Output(component_id="confirm-dialog", component_property="displayed"),
    Output(component_id="my-chart", component_property="figure"),
    Input(component_id="box_vs_scatter", component_property="value"),
    Input(component_id="x_feature", component_property="value"),
    Input(component_id="y_feature", component_property="value"),
    Input(component_id="color_dropdown", component_property="value"),
    Input(component_id="filter", component_property="value"),
)
def update_graph(box_vs_scatter, x_feature, y_feature, color, filter):
    df_input = df.loc[df["cell_group_in_pc"].isin(filter)]
    if box_vs_scatter == "box plot":

        fig = px.box(
            df_input,
            x=color,
            y=x_feature,
            color=color,
            hover_name="id",
            points="all",
        )
        return False, fig

    if box_vs_scatter == "scatterplot":
        fig = px.scatter(
            df_input,
            x=x_feature,
            y=y_feature,
            color=color,
            hover_name="id"
            #            scrollZoom = True,
        )

        return False, fig


if __name__ == "__main__":
    app.run_server(debug=True)
