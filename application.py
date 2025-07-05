from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd

from Dashboard import create_dashboard
from utils.data_processing import process_uploaded_file, process_for_export

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Healthcare Mortality Analysis"

app.layout = create_dashboard()

@app.callback(
    Output('tabs-content', 'children'),
    Output('download-dataframe-csv', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('btn-download', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def handle_callbacks(contents, filename, n_clicks, contents_export):
    trigger_id = ctx.triggered_id

    if trigger_id == 'upload-data' and contents:
        df_temp = process_uploaded_file(contents, filename)
        if isinstance(df_temp, str):
            return html.Div([df_temp]), no_update
        from Dashboard import generate_dashboard_content
        tabs = generate_dashboard_content(df_temp)
        return tabs, no_update

    elif trigger_id == 'btn-download' and contents_export:
        export_result = process_for_export(contents_export)
        if export_result:
            return no_update, export_result
        return no_update, None

    return no_update, no_update

if __name__ == '__main__':
    app.run(debug=True, port=8052)
