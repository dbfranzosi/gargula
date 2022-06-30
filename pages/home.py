import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


dash.register_page(__name__, path='/')

layout = html.Div(children=[
    dbc.Row([
        dbc.Col(html.A(html.Img(src=dash.get_asset_url("gargula-inverted.jpg"), height="300px"),href=dash.get_relative_path('/gargula'))),
        dbc.Col(html.Img(src=dash.get_asset_url("coruja-inverted.jpg"), height="300px"))
        ],                                
        align="center"
    ),
    dbc.Row([
        dbc.Col(dcc.Markdown(''' 
            *__You are in the dark.__*

            *__You understand almost nothing around you.__*
            
            *__And suddenly you die.__*  ''' )),
        dbc.Col(dcc.Markdown('''
            *__You need to be something else to see in the dark.__*
            
            *__You need much bigger eyes.__* '''))
        ],                                
        align="center"
    )
])