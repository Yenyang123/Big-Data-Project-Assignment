import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score, davies_bouldin_score
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

    dcc.Download(id="download-dataframe-csv"),

    dbc.Button("Export Cleaned Data", id="btn-download", color="primary", className="mb-3"),

    dbc.Tabs(id='tabs-content', children=[])
], fluid=True)

# Helper function to create a Dash DataTable from a DataFrame
def safe_table(dataframe):
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


    # Deaths Heatmap
    pivot = df.pivot_table(index='Cause Name', columns='Year', values='Deaths', aggfunc='sum')
    heatmap_fig = px.imshow(pivot, text_auto=True, aspect="auto", title="Heatmap of Deaths by Cause and Year")
    heat_stats = pivot.describe().round(2).reset_index()

    # K-Means Clustering Section
    yearly = df.groupby('Year')['Deaths'].sum().reset_index()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(yearly[['Year', 'Deaths']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    yearly['Cluster'] = kmeans.fit_predict(scaled)

    cluster_fig = px.scatter(yearly, x='Year', y='Deaths', color='Cluster', title="K-Means Clustering of Deaths Over Time")

    # Calculate clustering metrics
    silhouette = silhouette_score(scaled, yearly['Cluster'])
    davies = davies_bouldin_score(scaled, yearly['Cluster'])

    cluster_metrics = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
        'Value': [round(silhouette, 4), round(davies, 4)]
    })

    # Polynomial Regression
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    poly_preds = poly_model.predict(X)
    poly_fig = go.Figure()
    poly_fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'],
                                  mode='markers+lines', name='Actual Deaths'))
    poly_fig.add_trace(go.Scatter(x=yearly['Year'], y=poly_preds,
                                  mode='lines', name='Polynomial Fit',
                                  line=dict(color='purple')))
    poly_fig.update_layout(title="Polynomial Regression Fit (Degree 2)",
                           xaxis_title="Year", yaxis_title="Deaths")

    # Table of Actual vs Predicted Deaths (Polynomial)
    poly_results_df = pd.DataFrame({
    "Year": yearly['Year'],
    "Actual Deaths": yearly['Deaths'].astype(int),
    "Predicted Deaths": poly_preds.astype(int)
    })


    # Polynomial Regression Metrics
    r2_poly = r2_score(y, poly_preds)
    rmse_poly = mean_squared_error(y, poly_preds, squared=False)
    mae_poly = mean_absolute_error(y, poly_preds)

    poly_metrics_df = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [round(r2_poly, 4), round(rmse_poly, 2), round(mae_poly, 2)]
    })


    # Return all Tabs
    return [
        dbc.Tab(label='Top Causes by State', children=[
            dcc.Graph(figure=bar_fig),
            html.Br(), safe_table(top5_stats)
        ]),
        dbc.Tab(label='Heart Disease Trends', children=[
            dcc.Graph(figure=line_fig),
            html.Br(), safe_table(heart_summary)
        ]),
        dbc.Tab(label='Forecasting (Regression)', children=[
            dcc.Graph(figure=regression_fig),
            html.Br(), safe_table(pred_df),
            html.H5("Model Evaluation Metrics"), safe_table(lin_metrics_df)
        ]),
        dbc.Tab(label='Deaths Heatmap', children=[
            dcc.Graph(figure=heatmap_fig),
            html.Br(), safe_table(heat_stats)
        ]),
        dbc.Tab(label='K-Means Clustering', children=[
            dcc.Graph(figure=cluster_fig),
            html.Br(),
            safe_table(yearly),
            html.H5("Clustering Evaluation Metrics"),
            safe_table(cluster_metrics)
        ]),
        dbc.Tab(label='Polynomial Regression', children=[
            dcc.Graph(figure=poly_fig),
            html.Br(),
            html.H5("Actual vs Predicted Deaths (Polynomial Regression)"),
            safe_table(poly_results_df),
            html.Br(),
            html.H5("Model Evaluation Metrics"),
            safe_table(poly_metrics_df)
        ])
    ]

# Callback to handle file uploads and data export (CSV)
@app.callback(
    Output('tabs-content', 'children'),
    Output('download-dataframe-csv', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('btn-download', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_output(contents, filename, n_clicks, contents_export):
    trigger_id = ctx.triggered_id

    if trigger_id == 'upload-data' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df_temp = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            else:
                return html.Div(['This file type is not supported. Please upload a CSV file.']), no_update
        except Exception as e:
            return html.Div([f'There was an error processing this file: {e}']), no_update

        tabs = generate_dashboard_content(df_temp)
        return tabs, no_update

    elif trigger_id == 'btn-download' and contents_export:
        content_type, content_string = contents_export.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df_export = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Apply the same cleaning and mapping as in generate_dashboard_content
            df_export['Deaths'] = pd.to_numeric(df_export['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
            df_export['Age-adjusted Death Rate'] = pd.to_numeric(df_export['Age-adjusted Death Rate'], errors='coerce')
            df_export = df_export[df_export['State'] != 'United States'].copy()

            malaysian_states = [
                "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
                "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
                "Selangor", "Terengganu", "Kuala Lumpur", "Putrajaya", "Labuan"
            ]
            us_states = df_export['State'].unique()
            np.random.seed(42)
            state_map = dict(zip(us_states, np.random.choice(malaysian_states, size=len(us_states), replace=True)))
            df_export['State'] = df_export['State'].map(state_map)

            return no_update, dcc.send_data_frame(df_export.to_csv, "cleaned_mortality_data.csv", index=False)
        except Exception as e:
            print(f"Error during CSV export: {e}")
            return no_update, None

    return no_update, no_update

if __name__ == '__main__':
    app.run(debug=True, port=8052)