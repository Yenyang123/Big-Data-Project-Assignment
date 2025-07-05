from dash import dcc, html
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import pandas as pd
import numpy as np  # ✅ Make sure to import numpy
from utils.data_processing import safe_table

def polynomial_regression_tab(df):
    heart_df = df[df['Cause Name'] == 'Heart disease']
    yearly = heart_df.groupby('Year')['Deaths'].sum().reset_index()
    x = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Deaths'].values

    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(x, y)
    preds = model.predict(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=preds, mode='lines', name='Polynomial Fit'))

    # ✅ Fix: Manual RMSE calculation using np.sqrt
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    metrics = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [
            round(r2, 4),
            round(rmse, 2),
            round(mae, 2)
        ]
    })

    results = pd.DataFrame({
        'Year': yearly['Year'],
        'Actual': yearly['Deaths'].astype(int),
        'Predicted': preds.astype(int)
    })

    return [dbc.Tab(label='Polynomial Regression', children=[
        dcc.Graph(figure=fig),
        html.H5("Actual vs Predicted"), safe_table(results),
        html.H5("Model Evaluation Metrics"), safe_table(metrics)
    ])]
