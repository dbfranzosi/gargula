from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
import uuid
import os
import credentials
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

#from dbconnection import gargula_connection

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
#app.config.suppress_callback_exceptions = True
#app.server.secret_key = 'my_secret_key'
server.config['SECRET_KEY'] = 'mysecretkey'

# @app.server.before_request
# def before_request():
#     session.permanent = True
#     app.permanent_session_lifetime = timedelta(minutes=5)

# Connect to DB
db_server=os.environ['DB_SERVER']
db_name=os.environ['DB_NAME']
db_user=os.environ['DB_USER']
db_password=os.environ['DB_PASSWORD']

server.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+psycopg2://{}:{}@{}/{}".format(db_user,db_password, db_server, db_name)

db = SQLAlchemy(server)

#server.config['SECRET_KEY'] = 'mysecretkey'
#basedir = os.path.abspath(os.path.dirname(__file__))
#server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'test.db')
#server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#db = SQLAlchemy(server)
#Migrate(server,db)

with server.app_context():
    db.create_all()

# Define app layout

app.layout = html.Div([
	 dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href=app.get_relative_path("/"))),
                dbc.NavItem(dbc.NavLink("Gargula", href=app.get_relative_path("/gargula"))),
                dbc.NavItem(dbc.NavLink("Coruja", href=app.get_relative_path("/coruja"))),
                dbc.NavItem(dbc.NavLink("Info", href=app.get_relative_path("/info"))),
                dbc.NavItem(dbc.NavLink("Login", href=app.get_relative_path("/login"))),
            ]
        ),
	dash.page_container
])

#if __name__ == '__main__':
#	app.run_server(debug=True)

class User(db.Model):
    """An admin user capable of viewing reports.

    :param str email: email address of user
    :param str password: encrypted password for the user

    """
    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.email

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False