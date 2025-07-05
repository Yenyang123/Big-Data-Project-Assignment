from dash import dcc, html
import dash_bootstrap_components as dbc
from models.Linear_regression import linear_regression_tab
from models.Polynomial_regression import polynomial_regression_tab
from models.kmeans_clustering import kmeans_tab
import plotly.express as px
from utils.data_processing import safe_table

def create_dashboard():
    return dbc.Container([
        html.H1("Healthcare Mortality Analysis Dashboard (Malaysia)", className="text-center my-4"),

        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
            },
            multiple=False
        ),

        dbc.Label("Select Prediction Horizon (Years):", className="mt-3"),
        dcc.Dropdown(
            id='prediction-horizon',
            options=[
                {"label": "1 Year", "value": 1},
                {"label": "2 Years", "value": 2},
                {"label": "5 Years", "value": 5}
            ],
            value=1,
            clearable=False,
            style={'width': '200px', 'margin-bottom': '20px'}
        ),

        dcc.Download(id="download-dataframe-csv"),
        dbc.Button("Export Cleaned Data", id="btn-download", color="primary", className="mb-3"),

        dbc.Tabs(id='tabs-content', children=[])
    ], fluid=True)


def generate_dashboard_content(df, horizon=1):
    # Bar Plot Tab: Top Causes by State
    top5 = df.groupby(['State', 'Cause Name'])['Deaths'].sum().reset_index()
    top5 = top5.sort_values(['State', 'Deaths'], ascending=[True, False]).groupby('State').head(5)
    bar_fig = px.bar(top5, x='Deaths', y='Cause Name', color='State', barmode='group',
                     title="Top 5 Causes of Death by Malaysian State")
    top5_stats = top5.groupby('State')['Deaths'].sum().reset_index().rename(columns={'Deaths': 'Total Deaths'})

    bar_tab = dbc.Tab(label='Top Causes by State', children=[
        dcc.Graph(figure=bar_fig),
        safe_table(top5_stats)
    ])

    # Include model tabs with the selected horizon
    return [
        bar_tab,
        *linear_regression_tab(df, horizon),
        *polynomial_regression_tab(df, horizon),
        *kmeans_tab(df)  # No forecast needed for clustering
    ]
