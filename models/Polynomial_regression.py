from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_processing import safe_table

def polynomial_regression_tab(df, horizon=1, verbose=False, show_graph=False):
    heart_df = df[df['Cause Name'] == 'Heart disease']
    yearly = heart_df.groupby('Year')['Deaths'].sum().reset_index()
    x = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Deaths'].values

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(x, y)
    preds = model.predict(x)

    last_year = yearly['Year'].max()
    future_years = np.arange(last_year + 1, last_year + 1 + horizon).reshape(-1, 1)
    future_preds = model.predict(future_years)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=preds, mode='lines', name='Polynomial Fit'))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=future_preds, mode='lines+markers', name='Future Forecast', line=dict(dash='dot')))

    metrics = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [
            round(r2_score(y, preds), 4),
            round(np.sqrt(mean_squared_error(y, preds)), 2),
            round(mean_absolute_error(y, preds), 2)
        ]
    })

    results = pd.DataFrame({
        'Year': np.concatenate([yearly['Year'], future_years.flatten()]),
        'Actual': list(yearly['Deaths'].astype(int)) + [None]*len(future_years),
        'Predicted': list(preds.astype(int)) + list(future_preds.astype(int))
    })

    if verbose:
        print("\n==== Polynomial Regression Predictions ====")
        print(results.to_string(index=False))
        print("\n==== Model Evaluation Metrics ====")
        print(metrics.to_string(index=False))

    if show_graph:
        fig.show()

    return [dbc.Tab(label='Polynomial Regression', children=[
        dcc.Graph(figure=fig),
        html.H5("Actual vs Predicted"), safe_table(results),
        html.H5("Model Evaluation Metrics"), safe_table(metrics)
    ])]

# Standalone use
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Deaths'] = pd.to_numeric(df['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
        df['Age-adjusted Death Rate'] = pd.to_numeric(df['Age-adjusted Death Rate'], errors='coerce')
        df = df[df['State'] != 'United States'].copy()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == '__main__':
    df = load_data(r'C:\Users\HeeJetHow\PycharmProjects\Big-Data-Project-Assignment\death.csv')
    if df is not None:
        polynomial_regression_tab(df, horizon=5, verbose=True, show_graph=True)
    else:
        print("No data loaded.")
