from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], use_pages=True)

app.layout = html.Div([
	 dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href=app.get_relative_path("/"))),
                dbc.NavItem(dbc.NavLink("Gargula", href=app.get_relative_path("/gargula"))),
                dbc.NavItem(dbc.NavLink("Coruja", href=app.get_relative_path("/coruja"))),
            ]
        ),
	dash.page_container
])

if __name__ == '__main__':
	app.run_server(debug=True)