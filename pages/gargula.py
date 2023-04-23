''' Backend ''' 
from reality.biology import biology
from reality.geography import eden
from models.group.group import gargalo
from settings import *
import pandas as pd

''' Frontend '''
import dash
from dash import dcc, html, callback, ctx
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

# should not use global variables, instead use long_callbacks and sharing data
# https://dash.plotly.com/sharing-data-between-callbacks
# convert to dcc.Store

releaser = True
releaser_ext = False

passing_turn = False
simulating = False
saving = False

hv_display = 0

visualize_settings()

lst_groups = listdir('./data/groups/')
lst_groups = [name.split('.')[0] for name in lst_groups]

lst_bios = listdir('./data/biologies/')
lst_bios = [name.split('.')[0] for name in lst_bios]

lst_areas = listdir('./data/areas/')
lst_areas = [name.split('.')[0] for name in lst_areas]

cards_init = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Biology", className="card-title"),
                    dbc.Input(id="name_biology-input", type="text", value="homo-virtualis", debounce=True),
                    html.P(id="biology_info_init",
                        className="card-text",
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Area", className="card-title"),
                    dbc.Input(id="name_area-input", type="text", value="Eden", debounce=True),
                    html.P(id="area_info_init",
                        className="card-text",
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Group", className="card-title"),
                    dbc.Input(id="name_group-input", type="text", value="Gargalo", debounce=True),
                    html.P(id="group_info_init",
                        className="card-text",
                    ),
                ]
            )
        ),
    ], style={"margin-top": "15px", "margin-bottom": "15px"}
)

accordion_init = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(lst_bios, id="load_biology-list"), className="me-3",),
                        dbc.Col(dbc.Button(id="load_biology-buttom", children="Load biology", color="primary"), width="auto"),                 
                    ], style={"margin-top": "15px"}),                
                    dbc.Form(
                        dbc.Row([            
                                dbc.Label("Meiosis variation (%)", width="auto"),
                                dbc.Col(
                                    dbc.Input(id="meiosis-input", type="number", min=0, max=100, placeholder=10),
                                    className="me-1",
                                ),            
                                dbc.Col(dbc.Button(id="create_biology-buttom", children="Create biology", color="primary"), width="auto"),
                            ],
                            className="g-2",
                        ),  style={"margin-top": "15px"}    
                    ),
                    html.Div("", id="text_load_bio")           
                ],                
                title="Load or create biology",
            ),
            dbc.AccordionItem(
                [
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(lst_areas, id="load_area-list"), className="me-3",),
                        dbc.Col(dbc.Button(id="load_area-buttom", children="Load area", color="primary"), width="auto"),
                    ], style={"margin-top": "15px"}),                
                    dbc.Form(
                        dbc.Row([            
                                dbc.Label("Food", width="auto"),
                                dbc.Col(
                                    dbc.Input(id="food-input", type="number", min=1, max=25, placeholder=10),
                                    className="me-1",
                                ),            
                                dbc.Label("Food production", width="auto"),
                                dbc.Col(
                                    dbc.Input(id="food_production-input", type="number", min=1, max=25, placeholder=10),
                                    className="me-1",
                                ),            
                                dbc.Col(dbc.Button(id="create_area-buttom", children="Create area", color="primary"), width="auto"),
                            ],
                            className="g-2",
                        ),  style={"margin-top": "15px"}    
                    ),
                    html.Div("", id="text_load_area")           
                ],
                title="Load or create area",
            ),
            dbc.AccordionItem([                     
                    dbc.Row([
                        dbc.Col(dcc.Checklist(lst_groups, id="load_group-list", inline=True), className="me-3",),
                        dbc.Col(dbc.Button(id="clean_group-buttom", children="Clean group", color="primary"), width="auto"),
                        dbc.Col(dbc.Button(id="load_group-buttom", children="Load groups", color="primary"), width="auto"),
                    ], style={"margin-top": "15px"}),                
                    dbc.Form(
                        dbc.Row([            
                                dbc.Label("Number of homo-virtualis", width="auto"),
                                dbc.Col(
                                    dbc.Input(id="nrhv-input", type="number", min=1, max=25, value=10),
                                    className="me-1",
                                ),            
                                dbc.Col(dbc.Button(id="create_group-buttom", children="Create group", color="primary"), width="auto"),
                            ],
                            className="g-2",
                        ),  style={"margin-top": "15px"}    
                    ),
                    html.Div("", id="text_load_group")           
            ], title="Load or create group",
            ),
        ],
        start_collapsed=True,
    ), style={"margin-top": "15px", "margin-bottom": "15px"}
)

control_sim = html.Div([
        dcc.Interval(
            id='interval-component',
            interval=eden.timeunit*2000, # in milliseconds
            n_intervals=0
        ),
        html.Div([dbc.Button(id='sim-button', children="Run Simulation", n_clicks=0),], 
                className="d-grid gap-2", style={"margin-top": "15px"}),    
        html.Div([dbc.Button(id='savegroup-button', children="Save", n_clicks=0),], 
                className="d-grid gap-2", style={"margin-top": "15px", "margin-bottom": "15px"})
], style={"margin-top": "15px", "margin-bottom": "15px"})

family_tree = cyto.Cytoscape(
            id='cytoscape',        
            layout={'name': 'preset', 'animate': True},        
            #style={'width': '600px', 'height': '500px'},        
            stylesheet=[            
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'color': 'white',
                        'background-fit': 'cover',
                        'background-image': 'data(url)'
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

hv_panel =     html.Div([        
        html.Div(id='info_hv'),                
        dcc.Graph(id='fig_hv')
    ], style={"margin-top": "15px", "margin-bottom": "15px"})

group_panel = html.Div([        
        html.Div(id='info_area'),        
        html.Div(id='info_group'),         
        dbc.Row([
             dbc.Col(family_tree, width=6),
             dbc.Col(hv_panel, width=6)             
        ]),
        dcc.Graph(id='fig_gene') 
], style={"margin-top": "15px", "margin-bottom": "15px"})

layout = html.Div([
    cards_init,
    accordion_init,  
    control_sim,  
    group_panel    
],  style={"margin-left": "30px", "margin-right": "30px"})

@callback(
        Output('biology_info_init', 'children'),
        Output('area_info_init', 'children'),
        Output('group_info_init', 'children'),
        Output('text_load_bio', 'children'),
        Output('text_load_area', 'children'),        
        Output('text_load_group', 'children'),        
        Input('name_biology-input', 'value'),
        Input('name_area-input', 'value'),
        Input('name_group-input', 'value'),
        Input('load_biology-buttom', 'n_clicks'),                        
        Input('create_biology-buttom', 'n_clicks'),
        Input('load_area-buttom', 'n_clicks'),   
        Input('create_area-buttom', 'n_clicks'),             
        Input('load_group-buttom', 'n_clicks'),                
        Input('create_group-buttom', 'n_clicks'),
        Input('clean_group-buttom', 'n_clicks'),                
        State('load_biology-list', 'value'),        
        State('load_area-list', 'value'),        
        State('load_group-list', 'value'),        
        State('nrhv-input', 'value'),        
        )
def initialization(name_biology, name_area, name_group, n_load_bio, n_create_bio, n_load_area, n_create_area,
        n_load_group, n_create_group, n_clean_group, biology_list, area_list, group_list, nrhv):
    global gargalo, biology, eden
    global simulating, saving, passing_turn

    if (simulating or saving or passing_turn):
        PreventUpdate

    info_biology = ''
    info_area = ''
    info_group = ''
    text_load_bio = ''
    text_load_area = ''
    text_load_group = ''

    trigger = ctx.triggered_id

    if (trigger == 'name_biology-input'):
        if (name_biology in lst_bios):
            info_biology += '\n This species already exists. Choose another name.'
        else:
            biology.name = name_biology
        
    elif (trigger == 'name_area-input'):
        if (name_area in lst_areas):
            info_area += '\n This area already exists. Choose another name.'
        else:
            eden.name = name_area
    elif (trigger == 'name_group-input'):        
        if (name_group in lst_groups):
            info_group += '\n This group already exists. Choose another name.'
        else:
            gargalo.name = name_group
    elif(trigger == 'load_biology-buttom'):
        if (biology_list == None):
            text_load_bio = f"Choose a biology or group to load a biology."
            pass
        else:            
            biology = biology.load(biology_list)   
            gargalo.biology = biology             
            text_load_bio = f"Biology {biology_list} has been loaded." 
    elif(trigger == 'load_area-buttom'):
        if (area_list == None):
            text_load_area = f"Choose an area or group to load area."
            pass
        else:            
            eden = eden.load(area_list)  
            gargalo.home = eden              
            text_load_area = f"Area {area_list} has been loaded." 
    elif(trigger == 'load_group-buttom'):
        # Load group
        if (len(group_list) == 0 ):
            text_load_group = f"Choose a group to load or create a new group."
            pass
        elif (len(group_list) == 1):        
            group_name = group_list[0]        
            gargalo = gargalo.load(group_name)
            biology = gargalo.biology
            eden = gargalo.home       
            text_load_group = f"Group, biology and area from {group_name} have been loaded."   
        else:            
            group_name = group_list[0]        
            gargalo = gargalo.load(group_name)
            biology = gargalo.biology
            eden = gargalo.home 
            str_groups = group_name
            str_non_comp = ''
            
            for group_name in group_list[1:]:                
                gargalo, bmerge = gargalo.merge(group_name) 
                if bmerge:
                    str_groups += f"{group_name}" 
                else:
                    str_non_comp += f"{group_name}" 
            gargalo.name = str_groups
            eden.name = eden.name + str_groups
            print(gargalo.get_info())
            for hv in gargalo.hvs.values():
                print(hv.name)
                print(hv.id)
                print(hv.age)
                print(hv.group.name)
            text_load_group = f"Group, biology and area from {group_name} have been loaded. \n \
                Groups ({str_groups}) merged. Groups ({str_non_comp}) not compatible. \n \
                Name of group and area changed."   
            

    elif (trigger == 'create_biology-buttom'):
        if (name_biology in lst_bios):
            info_biology += '\n This group already exists. Choose another name.'
        else:
            # change the parameters of bio here
            pass

    elif (trigger == 'create_area-buttom'):
        if (name_area in lst_areas):
            info_area += '\n This area already exists. Choose another name.'
        else:
            #change parameters of area here
            pass

    elif (trigger == 'create_group-buttom'):
        if (name_group in lst_groups):
            info_group += '\n This group already exists. Choose another name.'
        else:
            gargalo.generate_gargalo(nrhv)

    elif (trigger == 'clean_group-buttom'):        
        gargalo = gargalo.clean()        

    info_biology += biology.get_info()    
    info_area += eden.get_info() 
    info_group += gargalo.get_info()    

    return info_biology, info_area, info_group, text_load_bio, text_load_area, text_load_group

@callback(Output('sim-button', 'color'),
        Output('sim-button', 'children'),
        Output('savegroup-button', 'disabled'),   
        Output('load_group-list', 'options'),          
        Input('sim-button', 'n_clicks'),
        Input('savegroup-button', 'n_clicks'),
        )
def control_sim(n_run, n_sim): 
    global simulating, passing_turn, saving

    lst_groups = listdir('./data/groups/')
    lst_groups = [name.split('.')[0] for name in lst_groups]        

    trigger = ctx.triggered_id
    if (trigger == 'savegroup-button'):
        saving = True
        biology.save()
        eden.save()
        gargalo.save()    
        gargalo.write_histories()
        saving = False

        lst_groups = listdir('./data/groups/')
        lst_groups = [name.split('.')[0] for name in lst_groups]        

    if (n_run % 2 == 0):        
        simulating = False
        return "primary", "Run Simulation", False, lst_groups
    else:
        simulating = True
        return "danger", "Stop Simulation", True, lst_groups

# Multiple components can update everytime interval gets fired.
@callback(Output('info_area', 'children'),
            Output('info_group', 'children'),            
            Output('fig_gene', 'figure'),
            Output('cytoscape', 'elements'), 
            Input('interval-component', 'n_intervals'))
def update_graph_live(n): 
    global simulating, passing_turn, saving    

    holder = (not simulating) or saving or passing_turn
    if holder:
        raise PreventUpdate 
    passing_turn = True
    #before = time.time()
    passing_turn = not eden.pass_day()       
    #after = time.time()    
    #print('interval=', after - before)    

    # Info
    info_area = eden.get_info() 
    info_group = gargalo.get_info()    

    # hvs        
    profiles = gargalo.get_profiles()           
    profiles = profiles[(profiles['energy']>0) & (profiles['energy_pool']>0)] 

    # Genetic survey
    data_gene = gargalo.history.get_genes()
    y_genes, y_traits, y_actions = gargalo.history.get_indicators()     
       
    fig_genes = make_subplots(rows=2, cols=2, 
                subplot_titles=["Averaged gen values", "Average gen evolution", "Averaged trait values", "Nr of actions"])
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
    #print(family)

    return info_area, info_group, fig_genes, family

@callback(  Output('info_hv', 'children'), 
            Output('fig_hv', 'figure'),                                   
            Input('cytoscape', 'tapNodeData'))
def update_hv_panel(cyto_data):       
    
    lst_ids = gargalo.get_list_ids()   
    # add list of hv to info
    info_hv = 'This homo-virtualis is not in the group.'
    fig_hv = make_subplots(rows=2, cols=2, 
                    subplot_titles=["gen values", "trait values", "Rewards", "Nr of actions"])     
    
    hv_display = -1
    if cyto_data:
        hv_display=int(cyto_data['id'])        
    
    if hv_display in lst_ids:           
        hv = gargalo.hvs[hv_display]    
        info_hv = hv.get_info(show_genes=False, show_action=True, show_visible=False)        
        genes = hv.get_genes()
        traits = hv.genes.phenotype.traits
        indicators = hv.history.get_indicators()
        y_actions = hv.history.get_counter()
        
        fig_hv.add_trace(go.Bar(y=genes, showlegend=False), row=1, col=1)
        fig_hv.add_trace(go.Bar(x=list(traits.values()), y=list(traits.keys()), showlegend=False, orientation='h'), row=1, col=2)        
        fig_hv.add_trace(go.Scatter(y=indicators.reward, mode="lines", showlegend=False), row=2, col=1)    
        for action in ACTIONS:                 
            fig_hv.add_trace(go.Scatter(y=y_actions[action], mode="lines", name=action), row=2, col=2)    
        #fig_hv.add_trace(go.Scatter(x=indicators.resistance, y=indicators.reward, mode="markers", showlegend=False), row=3, col=1)    

    # fig_hv.update_layout(
    # autosize=False,
    # width=1500,
    # height=800,
    # # margin=dict(
    # #     l=50,
    # #     r=50,
    # #     b=100,
    # #     t=100,
    # #     pad=4
    # # ),
    # #paper_bgcolor="LightSteelBlue",
    # )  

    return info_hv, fig_hv 