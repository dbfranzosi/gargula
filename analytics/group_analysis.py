from os.path import exists
import pandas as pd

def load_history(name):    
    if not exists(f'./data/history/{name}.csv'):
        return None
    with open(f'./data/history/{name}.csv', 'r') as f:
        df = pd.read_csv(f)
    return df

def load_hvs(name):            
    if not exists(f'./data/history/{name}_hvs.csv'):
        return None
    with open(f'./data/history/{name}_hvs.csv', 'r') as f:
        columns = ["id"] + list(range(200))
        df = pd.read_csv(f, names=columns, index_col="id")
    return df

