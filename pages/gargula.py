''' Backend ''' 
from reality.biology import biology
from reality.geography import eden
from models.group.group import gargalo
from settings import *
from models.group.visualization import *
import pandas as pd

''' Frontend '''
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash_cytoscape as cyto
from os import listdir
import time

dash.register_page(__name__)

releaser = True
releaser_ext = False

passing_turn = False
simulating = False
saving = False

visualize_settings()

lst_groups = listdir('./data/groups/')
lst_groups = [name.split('.')[0] for name in lst_groups]
print(lst_groups)
lst_bios = listdir('./data/biologies/')
lst_bios = [name.split('.')[0] for name in lst_bios]
lst_bios.append('New')
print(lst_bios)

layout = html.Div([
    html.H4("Load groups"),
    dbc.Row([
        dbc.Col(dcc.Checklist(lst_groups, id="load_group-list", inline=True), className="me-3",),
        dbc.Col(dbc.Button(id="load_group-buttom", children="Load groups", color="primary"), width="auto"),
    ]),
    html.Div(id='info_loadgroup'), 
    html.H4("Create group"),
    dbc.Form(
    dbc.Row(
        [
            dbc.Label("Group's name", width="auto"),
            dbc.Col(
                dbc.Input(id="namegroup-input", type="text", placeholder="Enter group's name"),
                className="me-3",
            ),
            dbc.Label("Number of homo-virtualis", width="auto"),
            dbc.Col(
                dbc.Input(id="nrhv-input", type="number", min=1, max=25, placeholder=10),
                className="me-3",
            ),            
            dbc.Label("Biology", width="auto"),
            dbc.Col(
                dcc.Dropdown(lst_bios, 'New', id='bios-dropdown'),                
                className="me-3",
            ),
            dbc.Col(dbc.Button(id="create_group-buttom", children="Create group", color="primary"), width="auto"),
        ],
        className="g-2",
    )    ),    
    html.Div(id='info_creategroup'),         
    html.Div([dbc.Button(id='sim-button', children="Run Simulation", n_clicks=0),], 
                className="d-grid gap-2", style={"margin-top": "15px"}),    
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=eden.timeunit*2000, # in milliseconds
            n_intervals=0
        ),
        # Constantly updated
        html.Div(id='info_area'),        
        html.Div(id='info_group'),         
        html.Div([dbc.Button(id='savegroup-button', children="Save Group", n_clicks=0),], 
                className="d-grid gap-2", style={"margin-top": "15px"}),        
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
                        'label': 'data(label)',
                        'color': 'white'
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
])


@callback(Output('info_loadgroup', 'children'),
        Input('load_group-buttom', 'n_clicks'),
        State('load_group-list', 'value'),
        )
def load_group(n, group_list):     
    global gargalo
    
    if (len(group_list) == 0 ):
        print("Choose a group to load.")
        PreventUpdate
    elif (len(group_list) == 1):        
        group_name = group_list[0]
        print('Loading '+ group_name)
        #gargalo.load(group_name)
        gargalo = gargalo.load(group_name)
        filename = f'./data/biologies/bio_{group_name}.pickle'
        biology.load(filename)
    else:
        print("Combining different groups into one. Not implemented yet.")
    
    # Info    
    info_group = 'Loaded '+gargalo.get_info()
    return info_group


@callback(Output('info_creategroup', 'children'),
        Input('create_group-buttom', 'n_clicks'),
        State('namegroup-input', 'value'),
        State('nrhv-input', 'value'),
        State('bios-dropdown', 'value'),        
        )
def create_group(n, name, nr, bio): 
    if (bio != "New"):
        filename = f'./data/biologies/{bio}.pickle'
        biology.load(filename)
    gargalo.name = name
    gargalo.generate_gargalo(nr)

    # Info    
    info_group = 'Created '+gargalo.get_info()
    return info_group


@callback(Output('sim-button', 'color'),
        Output('sim-button', 'children'),            
        Input('sim-button', 'n_clicks'),
        )
def control_sim(n): 
    global simulating
    if (n % 2 == 0):
        print("test")
        simulating = False
        return "primary", "Run Simulation"
    else:
        simulating = True
        return "danger", "Stop Simulation"

@callback(Output('savegroup-button', 'color'),
        Output('savegroup-button', 'children'),            
        Input('savegroup-button', 'n_clicks'),
        )
def save_group(n): 
    global simulating, passing_turn, saving
    if (simulating or passing_turn):
        raise PreventUpdate    
    saving = True
    gargalo.save()    
    saving = False
    # move to long_callback

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
@callback(Output('info_area', 'children'),
            Output('info_group', 'children'),
            Output('fig_hvs', 'figure'),
            Output('fig_gene', 'figure'),
            Output('cytoscape', 'elements'), 
            Input('interval-component', 'n_intervals'))
def update_graph_live(n): 
    global simulating, passing_turn, saving    
    holder = (not simulating) or saving or passing_turn
    if holder:
        raise PreventUpdate 
    passing_turn = True
    before = time.time()
    passing_turn = not eden.pass_day()       
    after = time.time()    
    print('interval=', after - before)    

    # Info
    info_area = eden.get_info() 
    info_group = gargalo.get_info()

    # hvs        
    profile = gargalo.get_profiles()           
    profile = profile[(profile['energy']>0) & (profile['energy_pool']>0)]        

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

    return info_area, info_group, fig_hvs, fig_genes, family

@callback(  Output('info_hv', 'children'), 
            Output('fig_hv', 'figure'),                       
            Input('input-hv', 'value'))
def update_hv_panel(hv_sel):     
    
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
        if (hv.history.__len__() == 0):
            y_actions = {action : 0.0 for action in ACTIONS}
        else:
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

    return info_hv, fig_hv 