import dash
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div([
    dcc.Markdown(''' 
    # Gargula

    ##### by Diogo Buarque Franzosi

    Gargula is a framework for the simulation of multi-species and multi-agent interactions.
    
    * The individuals are called **homo-virtualis (HV)**.
    * Each HV has a **genetics** which determines its **phenotype**.
    * Each HV has a **mind** that is used to decide **actions**.
    * Mind's **memories** are based on an interaction network [1].
    * Decision policy is based on a Deep Q Learning [2] algorithm.

    #### Bibliography

    * [1] IN
    * [2] DQL

    # Coruja

    Coruja means owl. It provides tools for the analysis of historical data from Gargula simulations.

    ''')
])
