import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Data from U.S. Congress, Joint Economic Committee, Social Capital Project.
df = pd.read_csv("../data/03_primary/cells_reformated.csv")
app = dash.Dash(__name__)

# ------------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.ConfirmDialog(
            id="confirm-dialog",
            displayed=False,
            message="Please choose Dropdown variables!",
        ),
        html.H1("Scatter Matrix of neuron_features", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="my-dropdown",
            options=[{"label": s, "value": s} for s in df.columns],
            value=["cellType"],
            multi=True,
        ),
        dcc.Dropdown(
            id="color_dropdown",
            options=["species", "cell_group_type", "cell_group_in_pc", "cellType"],
            value="cellType",
        ),
        dcc.Graph(id="my-chart", figure={}),
    ]
)


# ------------------------------------------------------------------------
@app.callback(
    Output(component_id="confirm-dialog", component_property="displayed"),
    Output(component_id="my-chart", component_property="figure"),
    Input(component_id="my-dropdown", component_property="value"),
    Input(component_id="color_dropdown", component_property="value"),
)
def update_graph(dpdn_val, color):
    if len(dpdn_val) > 0:
        fig = px.scatter_matrix(df, dimensions=dpdn_val, color=color, hover_name="id")
        fig.update_traces(
            diagonal_visible=False, showupperhalf=True, showlowerhalf=True
        )
        fig.update_layout(
            yaxis1={"title": {"font": {"size": 3}}},
            yaxis2={"title": {"font": {"size": 3}}},
            yaxis3={"title": {"font": {"size": 3}}},
            yaxis4={"title": {"font": {"size": 3}}},
            yaxis5={"title": {"font": {"size": 3}}},
            yaxis6={"title": {"font": {"size": 3}}},
            yaxis7={"title": {"font": {"size": 3}}},
            yaxis8={"title": {"font": {"size": 3}}},
        )
        fig.update_layout(
            xaxis1={"title": {"font": {"size": 3}}},
            xaxis2={"title": {"font": {"size": 3}}},
            xaxis3={"title": {"font": {"size": 3}}},
            xaxis4={"title": {"font": {"size": 3}}},
            xaxis5={"title": {"font": {"size": 3}}},
            xaxis6={"title": {"font": {"size": 3}}},
            xaxis7={"title": {"font": {"size": 3}}},
            xaxis8={"title": {"font": {"size": 3}}},
        )
        return False, fig

    if len(dpdn_val) == 0:
        return True, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
