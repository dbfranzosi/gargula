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
from dash.exceptions import PreventUpdate
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash_cytoscape as cyto
import time

holder = True
visualize_settings()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=eden.timeunit*1000, # in milliseconds
            n_intervals=0
        ),
        # Constantly updated
        html.H4(id='info_area'),        
        html.H4(id='info_group'),         
        dcc.Graph(id='fig_hvs'),                
        dcc.Graph(id='fig_gene')        
    ]),
    html.Div([
    html.P("Family tree:"),
    cyto.Cytoscape(
        id='cytoscape',        
        layout={'name': 'preset', 'animate': True},        
        style={'width': '600px', 'height': '500px'},        
        stylesheet=[            
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)'
                }
            },
            {
                'selector': '[role *= "mother"]',
                'style': {
                    'line-color': 'blue'
                }
            },
            {
                'selector': '[role *= "father"]',
                'style': {
                    'line-color': 'red'
                }
            }
        ]
    ),
    dcc.Dropdown(id='dropdown-hv', options=[0], value=0),
    dcc.Graph(id='fig_hv')
    ])
])

# Multiple components can update everytime interval gets fired.
@app.callback(Output('info_area', 'children'),
            Output('info_group', 'children'),
            Output('fig_hvs', 'figure'),
            Output('fig_gene', 'figure'),
            Output('cytoscape', 'elements'),            
            Input('interval-component', 'n_intervals'))
def update_graph_live(n): 
    global holder 
    #print(holder)
    if not holder:
        raise PreventUpdate 
    holder = False
    #before = time.time()
    holder = eden.pass_day()       
    #after = time.time()    
    #print('interval=', after - before)

    # Info
    info_area = eden.get_info() 
    info_group = gargalo.get_info()

    # hvs        
    profile = gargalo.get_profiles()           
    profile = profile[profile['energy']>0]          

    x = profile.index
    y = profile['age']
    colors = profile['power_attack']
    sz = profile['energy'] * 20 
    sy = profile['feature1'].astype('int')    
    text = profile['description']    

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
     text=text
    ))

    # Genetic survey
    data_gene = gargalo.history.get_genes()
    y_genes, y_traits, y_actions = gargalo.history.get_indicators()     
       
    fig_genes = make_subplots(rows=2, cols=2, 
                subplot_titles=["Averaged gen values", "", "Averaged trait values", "Nr of actions"])
    fig_genes.add_trace(go.Bar(y=data_gene, showlegend=False), row=1, col=1)
    #fig_genes.add_trace(go.Scatter(y=[2, 1, 3], mode="lines", showlegend=False), row=1, col=2)    
    for i in range(GEN_SIZE):
        fig_genes.add_trace(go.Scatter(y=y_genes[i], mode="lines", showlegend=False), row=1, col=2)
    for trait in TRAITS:                 
        fig_genes.add_trace(go.Scatter(y=y_traits[trait], mode="lines", name=trait), row=2, col=1)        
    for action in ACTIONS:                 
        fig_genes.add_trace(go.Scatter(y=y_actions[action], mode="lines", name=action), row=2, col=2)                

    # family
    family = gargalo.get_family()   

    return info_area, info_group, fig_hvs, fig_genes, family

@app.callback(Output('fig_hv', 'figure'),            
            Output('dropdown-hv', 'options'),
            Input('interval-component', 'n_intervals'),
            Input('dropdown-hv', 'value'))
def update_hv_graph(n, hv_sel):

    lst_ids = gargalo.get_list_ids()
    #print(lst_ids)
    if hv_sel not in lst_ids:    
        raise PreventUpdate 
    
    #print('hv_sel=', hv_sel)    
    hv = gargalo.hvs[hv_sel]    
    genes = hv.get_genes()
    traits = hv.genes.phenotype.traits
    #y_actions = hv.history.get_indicators()

    fig_hv = make_subplots(rows=2, cols=2, 
                subplot_titles=["gen values", "", "trait values", "Nr of actions"])
    fig_hv.add_trace(go.Bar(y=genes, showlegend=False), row=1, col=1)
    fig_hv.add_trace(go.Bar(x=list(traits.values()), y=list(traits.keys()), showlegend=False, orientation='h'), row=1, col=2)        
    # for action in ACTIONS:                 
    #     fig_hv.add_trace(go.Scatter(y=y_actions[action], mode="lines", name=action), row=2, col=1)     
    
    return fig_hv, lst_ids  


if __name__ == '__main__':    
    app.run_server(debug=True)

