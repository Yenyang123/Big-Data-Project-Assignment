from dash import dcc, html
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.data_processing import safe_table

def kmeans_tab(df):
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

    return [dbc.Tab(label='K-Means Clustering', children=[
        dcc.Graph(figure=fig),
        safe_table(yearly),
        html.H5("Clustering Evaluation Metrics"), safe_table(metrics)
    ])]
