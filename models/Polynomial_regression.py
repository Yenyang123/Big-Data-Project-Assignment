from dash import dcc, html
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import pandas as pd
from utils.data_processing import safe_table

def polynomial_regression_tab(df):
    heart_df = df[df['Cause Name'] == 'Heart disease']
    yearly = heart_df.groupby('Year')['Deaths'].sum().reset_index()
    X = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Deaths'].values

    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y)
    preds = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=preds, mode='lines', name='Polynomial Fit'))

    metrics = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [
            round(r2_score(y, preds), 4),
            round(mean_squared_error(y, preds, squared=False), 2),
            round(mean_absolute_error(y, preds), 2)
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
