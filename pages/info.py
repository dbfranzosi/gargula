import dash
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

dash.register_page(__name__)

with open('README.md', 'r') as f:
    s = f.read()

layout = html.Div([
    dcc.Markdown(s)
])
