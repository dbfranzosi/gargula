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
    * Decision policy is based on a Deep Reinforcement Learning (DRL) [2] algorithm. 
    * Learning happens during HV life span with DRL and by other HVs teachings.

    #### Bibliography

    * [1] Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu, Interaction Networks for Learning about Objects, Relations and Physics,  [arXiv:1612.00222], 
        Eric A. Moreno, Olmo Cerri, Javier M. Duarte, Harvey B. Newman, Thong Q. Nguyen, Avikar Periwal, Maurizio Pierini, Aidana Serikova, Maria Spiropulu, Jean-Roch Vlimant, JEDI-net: a jet identification algorithm based on interaction networks [arXiv:1908.05318]
    * [2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller, Playing Atari with Deep Reinforcement Learning, arXiv:1312.5602

    # Coruja

    Coruja provides tools for the analysis of historical data from Gargula simulations.

    
    ''')
])