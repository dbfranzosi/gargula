from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

# create directories not in repository (from gitignore)
from pathlib import Path
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./data/areas").mkdir(parents=True, exist_ok=True)
Path("./data/biologies").mkdir(parents=True, exist_ok=True)
Path("./data/groups").mkdir(parents=True, exist_ok=True)
Path("./data/history").mkdir(parents=True, exist_ok=True)
Path("./analytics/plots").mkdir(parents=True, exist_ok=True)
Path("./assets/avatars").mkdir(parents=True, exist_ok=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], use_pages=True)
# to create docker image use server (line below) and comment __main__ function
server = app.server

app.layout = html.Div([
	 dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href=app.get_relative_path("/"))),
                dbc.NavItem(dbc.NavLink("Gargula", href=app.get_relative_path("/gargula"))),
                dbc.NavItem(dbc.NavLink("Coruja", href=app.get_relative_path("/coruja"))),
                dbc.NavItem(dbc.NavLink("Info", href=app.get_relative_path("/info"))),
            ]
        ),
	dash.page_container
])

#if __name__ == '__main__':
#	app.run_server(debug=True)