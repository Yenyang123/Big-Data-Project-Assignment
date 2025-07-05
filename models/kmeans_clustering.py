from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.data_processing import safe_table

def kmeans_tab(df, verbose=False, show_graph=False):
    yearly = df.groupby('Year')['Deaths'].sum().reset_index()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(yearly[['Year', 'Deaths']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    yearly['Cluster'] = kmeans.fit_predict(scaled)

    fig = px.scatter(yearly, x='Year', y='Deaths', color='Cluster', title="K-Means Clustering of Deaths")

    metrics = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
        'Value': [
            round(silhouette_score(scaled, yearly['Cluster']), 4),
            round(davies_bouldin_score(scaled, yearly['Cluster']), 4)
        ]
    })

    if verbose:
        print("\n==== K-Means Clustering Results ====")
        print(yearly.to_string(index=False))
        print("\n==== Clustering Evaluation Metrics ====")
        print(metrics.to_string(index=False))

    if show_graph:
        fig.show()

    return [dbc.Tab(label='K-Means Clustering', children=[
        dcc.Graph(figure=fig),
        safe_table(yearly),
        html.H5("Clustering Evaluation Metrics"), safe_table(metrics)
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
        kmeans_tab(df, verbose=True, show_graph=True)
    else:
        print("No data loaded.")
