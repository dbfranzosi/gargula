''' Backend ''' 
from reality.biology import biology
from reality.geography import eden
from models.group.group import gargalo
from settings import *
from models.group.visualization import *
import pandas as pd

''' Frontend '''
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

visualize_settings()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4(id='info_area'),        
        html.H4(id='info_group'),         
        dcc.Graph(id='fig_hvs'),        
        html.H4('Genetic survey'),
        dcc.Graph(id='fig_gene'),        
        dcc.Interval(
            id='interval-component',
            interval=eden.timeunit*1000, # in milliseconds
            n_intervals=0
        )
    ])
)

# Multiple components can update everytime interval gets fired.
@app.callback(Output('info_area', 'children'),
            Output('info_group', 'children'),
            Output('fig_hvs', 'figure'),
            Output('fig_gene', 'figure'),
            Input('interval-component', 'n_intervals'))
def update_graph_live(n):        
    eden.pass_day()       

    # Info
    info_area = eden.get_info() 
    info_group = gargalo.get_info()

    # hvs
    hvs_repr = gargalo.get_repr()    
    hvs_names = gargalo.get_names()
    
    x = hvs_repr.index
    y = hvs_repr['age']
    colors = hvs_repr['power_attack']
    sz = hvs_repr['energy'] * 30
    sy = hvs_repr['feature1'].astype('int')    
    names = hvs_names['names']
    print(names)

    fig_hvs = go.Figure()
    fig_hvs.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers+text",
    #mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis",
        symbol=sy
    ),
     text=names
    ))

    # Genetic survey
    data_gene = gargalo.get_genes()
    y_genes, y_traits = gargalo.get_indicators()     
       
    fig_genes = make_subplots(rows=2, cols=2, subplot_titles=["Averaged gen values", "", "Averaged gen values", "Averaged trait values"])
    fig_genes.add_trace(go.Bar(y=data_gene, showlegend=False), row=1, col=1)
    fig_genes.add_trace(go.Scatter(y=[2, 1, 3], mode="lines", showlegend=False), row=1, col=2)    
    for i in range(GEN_SIZE):
        fig_genes.add_trace(go.Scatter(y=y_genes[i], mode="lines", showlegend=False), row=2, col=1)
    for trait in TRAITS:                 
        fig_genes.add_trace(go.Scatter(y=y_traits[trait], mode="lines", showlegend=False), row=2, col=2)        

    return info_area, info_group, fig_hvs, fig_genes
    #return info_area, info_group, fig_genes

if __name__ == '__main__':
    app.run_server(debug=True)

