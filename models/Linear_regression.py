from dash import dcc, html
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_processing import safe_table

def linear_regression_tab(df, horizon=5, verbose=False, show_graph=False):
    heart_df = df[df['Cause Name'] == 'Heart disease']
    yearly = heart_df.groupby('Year')['Deaths'].sum().reset_index()
    x = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Deaths'].values

    model = LinearRegression()
    model.fit(x, y)

    last_year = yearly['Year'].max()
    future_years = np.arange(last_year + 1, last_year + 1 + horizon).reshape(-1, 1)
    preds = model.predict(future_years)

    fitted_preds = model.predict(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Deaths'], mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=fitted_preds, mode='lines', name='Best Fit Line'))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=preds, mode='lines+markers', name='Forecast', line=dict(dash='dash')))

    metrics = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Value": [
            round(r2_score(y, fitted_preds), 4),
            round(mean_squared_error(y, fitted_preds) ** 0.5, 2),
            round(mean_absolute_error(y, fitted_preds), 2)
        ]
    })

    pred_table = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Deaths': preds.astype(int)})

    # Standalone mode outputs
    if verbose:
        print("\n==== Future Predictions ====")
        print(pred_table.to_string(index=False))
        print("\n==== Model Evaluation Metrics ====")
        print(metrics.to_string(index=False))

    if show_graph:
        fig.show()

    return [dbc.Tab(label='Forecasting (Regression)', children=[
        dcc.Graph(figure=fig),
        html.H5("Predicted Deaths"), safe_table(pred_table),
        html.H5("Model Evaluation Metrics"), safe_table(metrics)
    ])]

# Standalone: Load Data from file
def load_data(filepath=None):
    if filepath:
        try:
            df = pd.read_csv(filepath)
            df['Deaths'] = pd.to_numeric(df['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
            df['Age-adjusted Death Rate'] = pd.to_numeric(df['Age-adjusted Death Rate'], errors='coerce')
            df = df[df['State'] != 'United States'].copy()
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    return None

if __name__ == '__main__':
    df = load_data(filepath=r'C:\Users\HeeJetHow\PycharmProjects\Big-Data-Project-Assignment\death.csv')
    if df is not None:
        linear_regression_tab(df, horizon=5, verbose=True, show_graph=True)
    else:
        print("No data loaded.")
