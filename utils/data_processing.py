import pandas as pd
import base64
import io
from dash import dcc
import dash_table

def process_uploaded_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df['Deaths'] = pd.to_numeric(df['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
            df['Age-adjusted Death Rate'] = pd.to_numeric(df['Age-adjusted Death Rate'], errors='coerce')
            df = df[df['State'] != 'United States'].copy()
            return df
        else:
            return 'Unsupported file type. Please upload a CSV file.'
    except Exception as e:
        return f'Error processing file: {e}'

def process_for_export(contents):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return dcc.send_data_frame(df.to_csv, "cleaned_mortality_data.csv", index=False)
    except:
        return None

def safe_table(df):
    return dash_table.DataTable(
        columns=[{"name": str(i), "id": str(i)} for i in df.columns],
        data=df.to_dict('records'),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )
