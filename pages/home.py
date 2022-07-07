import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


dash.register_page(__name__, path='/')

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.A(html.Img(src=dash.get_asset_url("gargula-inverted.jpg"), width="250px"),href=dash.get_relative_path('/gargula')),
            dcc.Markdown(''' 
            *__You are in the dark.__*

            *__You understand almost nothing around you.__*
            
            *__And suddenly you die.__*  ''' ),
        ]),
        dbc.Col([
            html.A(html.Img(src=dash.get_asset_url("coruja-inverted.jpg"), width="250px"),href=dash.get_relative_path('/coruja')),
            dcc.Markdown('''
            *__You need to be something else to see in the dark.__*
            
            *__You need much bigger eyes.__* ''')])
        ],                                
        justify="center", align="center", className="h-50"
    ),
], style={'width': '800px',"height": "100vh"})