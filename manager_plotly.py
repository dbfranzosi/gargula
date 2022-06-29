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
            interval=eden.timeunit*2000, # in milliseconds
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
        )
    ]),
    # Individual HV panel
    html.Div([
        dcc.Input(id="input-hv", type="number", placeholder="Id of the hv", debounce=True),
        html.H4(id='info_hv'),        
        #dcc.Dropdown(id='dropdown-hv', options=[0], value=0),
        dcc.Graph(id='fig_hv')
    ]),
    html.Div([
        dcc.Dropdown(id='dropdown_metrics',
        options=["energy_pool","food_consumption","power_attack","resistance_attack","reward_eat","reward_rest","reward_sex","reward_violence","feature1"],
        #value='energy_pool'
        ),
        #html.H4(id='metrics'),        
        #dcc.Dropdown(id='dropdown-hv', options=[0], value=0),
        dcc.Graph(id='fig_metrics')
    ])    
])

def get_physical_rep(profile):
    x = profile.index
    y = profile['age']
    size =  profile['energy_pool'] * 10
    body = [x, y, {'color' : profile['energy']/10, 
            'size' : size,
            'symbol' : "pentagon",
            'opacity' : 0.6,
            'colorscale' : "Mint",
            'cmin':0, 'cmax':1 }]
    head = [x, y+size*0.5, {'color' : profile['reward_sex']/10, 
            'size' : size/3,
            'symbol' : "diamond",
            'opacity' : 1.,
            'colorscale' : "Hot",
            'cmin':0, 'cmax':1  } ]
    legs = [x, y-size*0.8, {'color' : profile['power_attack']/10, 
            'size' : size/3,
            'symbol' : "y-up",
            'opacity' : 1.,
            'colorscale' : "Sunset",
            'line':dict(
                color=profile['power_attack']/10,
                width=size/10
            ),
            'cmin':0, 'cmax':1  } ]
    return body, head, legs

# Multiple components can update everytime interval gets fired.
@app.callback(Output('info_area', 'children'),
            Output('info_group', 'children'),
            Output('fig_hvs', 'figure'),
            Output('fig_gene', 'figure'),
            Output('cytoscape', 'elements'), 
            Output('info_hv', 'children'), 
            Output('fig_hv', 'figure'),             
            Input('interval-component', 'n_intervals'),
            Input('input-hv', 'value'))
def update_graph_live(n, hv_sel): 
    global holder 
    print(holder)
    if not holder:
        raise PreventUpdate 
    holder = False
    before = time.time()
    holder = eden.pass_day()       
    after = time.time()    
    print('interval=', after - before)    

    # Info
    info_area = eden.get_info() 
    info_group = gargalo.get_info()

    # hvs        
    profile = gargalo.get_profiles()           
    profile = profile[(profile['energy']>0) & (profile['energy_pool']>0)]        

    # x = profile.index
    # y = profile['age']
    # # colors = profile['power_attack']
    # sz = profile['energy'] * 20 
    # sy = profile['feature1'].astype('int')    
    # text = profile['description']  
    body, head, legs = get_physical_rep(profile)

    fig_hvs = go.Figure()
    fig_hvs.add_trace(go.Scatter(
        x=body[0],
        y=body[1],
        mode="markers",
        marker=go.scatter.Marker(**body[2]), showlegend=False
    ))
    fig_hvs.add_trace(go.Scatter(
        x=head[0],
        y=head[1],
        mode="markers",
        marker=go.scatter.Marker(**head[2]), showlegend=False
    ))
    fig_hvs.add_trace(go.Scatter(
        x=legs[0],
        y=legs[1],
        mode="markers",
        marker=go.scatter.Marker(**legs[2]), showlegend=False
    ))
    fig_hvs.update_yaxes(range=[-50, 300])      
    fig_hvs.update_layout(    
        xaxis_title="Id",
        yaxis_title="Age",          
    )

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

    # Hv panel
    lst_ids = gargalo.get_list_ids()   
    # add list of hv to info
    info_hv = 'This homo-virtualis is not in the group.'
    fig_hv = make_subplots(rows=2, cols=2, 
                    subplot_titles=["gen values", "trait values", "", "Nr of actions"])     
    
    #print('hv_sel=', hv_sel)  
    if hv_sel in lst_ids:           
        hv = gargalo.hvs[hv_sel]    
        info_hv = hv.get_info(show_genes=False, show_action=True, show_visible=False)        
        genes = hv.get_genes()
        traits = hv.genes.phenotype.traits
        y_actions = hv.history.get_indicators()
        #print('y_actions=', y_actions)
        
        fig_hv.add_trace(go.Bar(y=genes, showlegend=False), row=1, col=1)
        fig_hv.add_trace(go.Bar(x=list(traits.values()), y=list(traits.keys()), showlegend=False, orientation='h'), row=1, col=2)        
        for action in ACTIONS:                 
            fig_hv.add_trace(go.Scatter(y=y_actions[action], mode="lines", name=action), row=2, col=2)    

    fig_hv.update_layout(
    autosize=False,
    width=1500,
    height=800,
    # margin=dict(
    #     l=50,
    #     r=50,
    #     b=100,
    #     t=100,
    #     pad=4
    # ),
    #paper_bgcolor="LightSteelBlue",
    )  

    return info_area, info_group, fig_hvs, fig_genes, family, info_hv, fig_hv 

# @app.callback(
#     Output('fig_metrics', 'figure'),      
#     Input('dropdown_metrics', 'value')
# )
# def update_metrics(value):
#     df = gargalo.history.load()
#     if (df == None):
#         raise PreventUpdate

#     pd.options.plotting.backend = "plotly"    
#     fig = df.plot(x=value)
#     return fig

if __name__ == '__main__':    
    app.run_server(debug=True)

