import dash
from dash import html, dcc, callback, Input, Output, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from settings import *
import analytics.group_analysis as gana

import pandas as pd
pd.options.plotting.backend = "plotly"

from os import listdir

dash.register_page(__name__)

lst_groups = listdir('./data/groups/')
lst_groups = [name.split('.')[0] for name in lst_groups]

group_panel = html.Div([  
        #dcc.Checklist(options=lst_groups, id="coruja_load_group-list", inline=True, value=["Gargalo"]),    
        dcc.Dropdown(options=lst_groups, id="coruja_load_group-list"),          
        html.Div(id='coruja_info_group'),  
        html.Div('Averaged traits over population'),                
        dbc.Row([
            dbc.Col(dcc.Graph(id='coruja_fig_traits1'), width=6),
            dbc.Col(dcc.Graph(id='coruja_fig_traits2'), width=6)            
        ]),                      
    ], style={"margin-top": "15px", "margin-bottom": "15px"})

hv_panel = html.Div([        
        dcc.Dropdown(options=[0], id="coruja_load_hv-list"),    
        html.Div(id='coruja_info_hv'),    
        dbc.Row([
            dbc.Col(dcc.Graph(id='coruja_fig_hv_info'), width=4),
            dbc.Col(dcc.Graph(id='coruja_fig_hv_actions'), width=4),
            dbc.Col(dcc.Graph(id='coruja_fig_hv_actions2'), width=4)
        ]),              
        dbc.Row([
            dbc.Col(dcc.Graph(id='coruja_fig_hv_reward'), width=6),
            dbc.Col(dcc.Graph(id='coruja_fig_hv_reward2'), width=6),
        ]),              
        dbc.Row([
            dbc.Col(dcc.Graph(id='coruja_fig_hv_targeted'), width=6),
            dbc.Col(dcc.Graph(id='coruja_fig_hv_targeted2'), width=6),
        ]),              
    # plot_targeted(ihv, 20)
    # plot_targeted(ihv, 50)
    # plot_targeted_scatter(ihv)
    ], style={"margin-top": "15px", "margin-bottom": "15px"})

layout = html.Div(children=[        
    group_panel,    
    hv_panel
])

df_group, df_hvs, df_info = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
ihvs = []

@callback(  Output('coruja_info_group', 'children'), 
            Output('coruja_fig_traits1', 'figure'),                       
            Output('coruja_fig_traits2', 'figure'),
            Output("coruja_load_group-list", 'options'),
            Output('coruja_info_hv', 'children'), 
            Output('coruja_fig_hv_info', 'figure'),                       
            Output('coruja_fig_hv_actions', 'figure'),
            Output('coruja_fig_hv_actions2', 'figure'),
            Output('coruja_fig_hv_reward', 'figure'),
            Output('coruja_fig_hv_reward2', 'figure'),
            Output('coruja_fig_hv_targeted', 'figure'),
            Output('coruja_fig_hv_targeted2', 'figure'),
            Output("coruja_load_hv-list", 'options'),
            Input("coruja_load_group-list", 'value'),
            Input("coruja_load_hv-list", 'value'))
def update_hv_panel(group_sel, hv_sel):

    global df_group, df_hvs, df_info, ihvs
    trigger = ctx.triggered_id
    
    if (group_sel == None):
        print("Group is None")
        raise PreventUpdate

    if (trigger == 'coruja_load_group-list'):      

        info_hv, fig_hv_info, fig_hv_actions, fig_hv_actions2 = dash.no_update, dash.no_update, dash.no_update, dash.no_update  
        fig_hv_reward, fig_hv_reward2, fig_hv_targeted, fig_hv_targeted2 = dash.no_update, dash.no_update, dash.no_update, dash.no_update

        lst_groups = listdir('./data/groups/')
        lst_groups = [name.split('.')[0] for name in lst_groups]

        print(f"Loading {group_sel}")

        df_group, df_hvs, df_info = gana.load_history(group_sel)

        info = f'Group {group_sel}...'

        fig_genetics = df_group.plot(y=['energy_pool','food_consumption','power_attack','resistance_attack', 'feature1'])
        fig_learning = df_group.plot(y=['reward_eat','reward_rest','reward_sex','reward_violence'])

        # HVs

        ihvs = list(df_hvs.xs('reward', level='Profile', axis=1).columns)
        ihvs.sort()       

    if (trigger == 'coruja_load_hv-list'):

        info, fig_learning, fig_genetics, lst_groups = dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if (hv_sel not in ihvs):
            info_hv = f'HV{hv_sel} not in group'
            fig_hv_info, fig_hv_actions, fig_hv_actions2 = dash.no_update, dash.no_update, dash.no_update
            fig_hv_reward, fig_hv_reward2, fig_hv_targeted, fig_hv_targeted2 = dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            dfs = gana.get_dfs(df_hvs, hv_sel)

            info_hv = f'Loading HV{hv_sel}'
            #fig_hv_info = gana.plot_profile(df_info, hv_sel)
            cols = df_info.columns.drop(['birth', 'age'])
            fig_hv_info = px.bar(df_info[cols].loc[hv_sel], labels={'index' : 'traits'}, title='HV Info')
            fig_hv_info.layout.update(showlegend=False)
            
            df_cnt = dfs.groupby(by=["action_name"]).count()[["reward"]].rename(columns={"reward":"count"}).reset_index()    
            fig_hv_actions = px.pie(df_cnt, values='count', names='action_name', title='HV action counter')

            df_reward = dfs.groupby(by=["action_name"]).mean()[["reward"]].rename(columns={"reward":"mean"}).reset_index()
            fig_hv_actions2 = px.pie(df_reward, values='mean', names='action_name', title='HV action mean')

            fig_hv_reward = px.scatter(dfs.loc[0:200], x="day", y="reward", trendline="ols")
            fig_hv_reward2 = px.scatter(dfs, x="day", y="reward", trendline="ols")

            max_hv = 15            
            dtmp = dfs[(dfs['action_name']=='attack')|(dfs['action_name']=='sex')]
            dtmp = dtmp[dtmp['target']<max_hv]

            dattack = dfs[(dfs['action_name']=='attack')&(dfs['target']<max_hv)]['target']
            dsex = dfs[(dfs['action_name']=='sex')&(dfs['target']<max_hv)]['target']

            ihvs_max = dtmp['target'].unique()            
                        
            fig_hv_targeted = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hv_targeted.add_trace(go.Histogram(x=dattack, name='attack', marker_color='magenta'), secondary_y=False)
            fig_hv_targeted.add_trace(go.Histogram(x=dsex, name='sex', marker_color='cyan'), secondary_y=False)

            pow_o_res = [dtmp[dtmp['target']==tgt].iloc[0]['pow_o_res'] for tgt in ihvs_max]
            colors = ['red' if pr < 0 else 'blue' for pr in pow_o_res]
            widths = [0.5, ]*len(ihvs_max)
            fig_hv_targeted.add_trace(go.Bar(x = ihvs_max, y = pow_o_res, marker_color=colors, width=widths, showlegend=False), secondary_y=True)
            
            fig_hv_targeted.update_layout(barmode='stack')
            fig_hv_targeted.update_yaxes(title_text="# targeted actions", secondary_y=False)
            fig_hv_targeted.update_yaxes(title_text="(pow-res)/(pow+res)", secondary_y=True)

            fig_hv_targeted2 = px.scatter(dtmp[(dtmp['action_name']=='attack')|(dtmp['action_name']=='sex')],
                x="pow_o_res", y='reward',
                color="action_name", opacity=0.2)

    return info, fig_learning, fig_genetics, lst_groups, info_hv, fig_hv_info, fig_hv_actions, fig_hv_actions2, \
        fig_hv_reward, fig_hv_reward2,  fig_hv_targeted, fig_hv_targeted2, ihvs
    
