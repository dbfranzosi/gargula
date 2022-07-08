import dash
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from settings import *
from analytics.group_analysis import *

import pandas as pd
pd.options.plotting.backend = "plotly"

from os import listdir

dash.register_page(__name__)

lst_groups = listdir('./data/groups/')
lst_groups = [name.split('.')[0] for name in lst_groups]

group_panel = html.Div([        
        html.Div(id='coruja_info_group'),  
        html.Div('Learning metrics'),                
        dcc.Graph(id='coruja_fig_learning'),
        html.Div('Ganetic metrics'),    
        dcc.Graph(id='coruja_fig_genetics')
    ], style={"margin-top": "15px", "margin-bottom": "15px"})

layout = html.Div(children=[    
    dcc.Checklist(lst_groups, id="coruja_load_group-list", inline=True, value=["Gargalo"]),    
    group_panel
])

@callback(  Output('coruja_info_group', 'children'), 
            Output('coruja_fig_learning', 'figure'),                       
            Output('coruja_fig_genetics', 'figure'),
            Input("coruja_load_group-list", 'value'))
def update_hv_panel(group_sel):

    if (group_sel == None):
        print("Group is None")
        raise PreventUpdate

    print(f"Loading {group_sel}")

    fig_learning = make_subplots(rows=2, cols=2, 
                    subplot_titles=["", "", "", ""])     
    fig_genetics = make_subplots(rows=1, cols=1, 
                    subplot_titles=["", "", "", ""])         

    hist, hist_hvs = {}, {}
    hist_hvs_long = {}
    info = ''    

    for name in group_sel:  
        h = load_history(name)
        hhv = load_hvs(name)

        hhv["age"] = hhv.count(axis=1)
        hhv["sum"] = hhv.sum(axis=1)        
        hhv["avg"] = hhv["sum"]/hhv["age"]
        
        hhv["slope"] = - hhv.apply(lambda x: x[0:int(x['age']/2)].sum(),axis=1)
        hhv["slope"] += hhv.apply(lambda x: x[int(x['age']/2):int(x['age'])].sum(),axis=1)
                
        hist_hvs_long[name] = hhv.loc[hhv[199].notna()]

        info += f'{name}: In {h.shape[0]} days {hhv.shape[0]} hvs died. {hist_hvs_long[name].shape[0]} lived more than 200 days. \n'

        fig_learning.add_trace(
            go.Bar(name='avg',
                x=hhv.index,
                y=hhv["avg"]),
            row=1, col=1)
        fig_learning.add_trace(
            go.Scatter(name='ageXslope',
                x=hhv["age"],
                y=hhv["slope"], mode="markers"),
            row=1, col=2)
        fig_learning.add_trace(
            go.Bar(name='age',
                x=hhv.index,
                y=hhv["age"]),
            row=2, col=1)
        # for action in ACTIONS:
        #     fig_learning.add_trace(
        #         go.Scatter(name=action,
        #             x=h.index,
        #             y=h[action], mode="lines"),
        #         row=2, col=2)
        
        
        print(h.columns)

        for trait in TRAITS:
            fig_genetics.add_trace(
                go.Scatter(name=trait,
                    x=h.index,
                    y=h[trait], mode="lines"),
                row=1, col=1)
        

    return info, fig_learning, fig_genetics
    


