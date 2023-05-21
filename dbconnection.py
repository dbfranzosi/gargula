import os
import sqlalchemy as sql
import credentials
import uuid
from flask_sqlalchemy import SQLAlchemy


def gargula_connection():
    """
    establish a connection to database
    """
    server=os.environ['DB_SERVER']
    database=os.environ['DB_NAME']
    user=os.environ['DB_USER']
    password=os.environ['DB_PASSWORD']
    
    engine = sql.create_engine("postgresql+psycopg2://{}:{}@{}/{}".format(user,password,server, database),echo=True)
    return engine