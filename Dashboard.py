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

def generate_dashboard_content(df_data):
    global df
    df = df_data.copy()
    df['Deaths'] = pd.to_numeric(df['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
    df['Age-adjusted Death Rate'] = pd.to_numeric(df['Age-adjusted Death Rate'], errors='coerce')
    df = df[df['State'] != 'United States'].copy()

    malaysian_states = [
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
        "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
        "Selangor", "Terengganu", "Kuala Lumpur", "Putrajaya", "Labuan"
    ]
    us_states = df['State'].unique()
    np.random.seed(42)
    state_map = dict(zip(us_states, np.random.choice(malaysian_states, size=len(us_states), replace=True)))
    df['State'] = df['State'].map(state_map)

    # Top Causes by State
    top5 = df.groupby(['State', 'Cause Name'])['Deaths'].sum().reset_index()
    top5 = top5.sort_values(['State', 'Deaths'], ascending=[True, False]).groupby('State').head(5)
    bar_fig = px.bar(top5, x='Deaths', y='Cause Name', color='State', barmode='group',
                     title="Top 5 Causes of Death by Malaysian State")
    top5_stats = top5.groupby('State')['Deaths'].sum().reset_index()
    top5_stats.rename(columns={'Deaths': 'Total Deaths in Top 5 Causes'}, inplace=True)

    # Heart Disease Trends
    heart_df = df[df['Cause Name'] == 'Heart disease']
    line_fig = px.line(heart_df, x='Year', y='Deaths', color='State', title="Heart Disease Deaths Over Time")
    heart_summary = heart_df.groupby('State')['Deaths'].mean().reset_index(name='Average Deaths')

    # Forecasting (Linear Regression)
    yearly = heart_df.groupby('Year')['Deaths'].sum().reset_index()
    X = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Deaths'].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array([2025, 2026, 2027, 2028, 2029]).reshape(-1, 1)
    predictions = model.predict(future_years)
    predicted_historical_deaths = model.predict(X)
    combined_years = np.concatenate((yearly['Year'].values, future_years.ravel()))
    combined_predicted_values = np.concatenate((predicted_historical_deaths, predictions))

    regression_fig = go.Figure()
    regression_fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'],
                                        mode='lines+markers', name='Actual Deaths'))
    regression_fig.add_trace(go.Scatter(x=combined_years, y=combined_predicted_values,
                                        mode='lines', name='Regression Fit & Forecast',
                                        line=dict(color='green', dash='dash')))
    regression_fig.add_trace(go.Scatter(x=future_years.ravel(), y=predictions,
                                        mode='markers', name='Predicted Future Deaths',
                                        marker=dict(symbol='star', size=8, color='orange')))
    regression_fig.update_layout(title="Linear Regression Forecast for Heart Disease Deaths",
                                 xaxis_title="Year", yaxis_title="Deaths")
    pred_df = pd.DataFrame({'Year': future_years.ravel(), 'Predicted Deaths': predictions.astype(int)})

    # Linear Regression Metrics
    r2_lin = r2_score(y, predicted_historical_deaths)
    rmse_lin = mean_squared_error(y, predicted_historical_deaths, squared=False)
    mae_lin = mean_absolute_error(y, predicted_historical_deaths)

    lin_metrics_df = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [round(r2_lin, 4), round(rmse_lin, 2), round(mae_lin, 2)]
    })


