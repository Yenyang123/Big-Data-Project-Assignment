import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table, no_update
import dash_bootstrap_components as dbc
import base64
import io

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global DataFrame
df = pd.DataFrame()

# Define the layout of the Dash application
app.layout = dbc.Container([
    # Page title
    html.H1("Healthcare Mortality Analysis Dashboard (Malaysia)", className="text-center my-4"),

    # Component for uploading a CSV file
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ', html.A('Select a CSV File')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
            'margin-bottom': '20px'
        },
        multiple=False
    ),

    # Component to trigger file download (for cleaned data)
    dcc.Download(id="download-dataframe-csv"),

    # Button to export the cleaned data
    dbc.Button("Export Cleaned Data", id="btn-download", color="primary", className="mb-3"),

    # Tabs component to display different analysis sections
    dbc.Tabs(id='tabs-content', children=[])
], fluid=True)

# Helper function to create a Dash DataTable from a DataFrame
def safe_table(dataframe):
    """
    Creates a Dash DataTable with columns and data from the given DataFrame.
    Includes basic styling for responsiveness and readability.
    """
    return dash_table.DataTable(
        columns=[{"name": str(i), "id": str(i)} for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )


