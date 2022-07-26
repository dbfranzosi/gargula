from os.path import exists
import pandas as pd
from settings import *

import seaborn as sns
import matplotlib.pyplot as plt


def load_history(name):    
    if not exists(f'./data/history/{name}.csv'):
        return None, None, None
    with open(f'./data/history/{name}.csv', 'r') as f:
        #df = pd.read_csv(f)
        columns = list(range(GEN_SIZE)) + TRAITS + ACTIONS
        df_group = pd.read_csv(f, names=columns, header=None)
        df_group['day'] = df_group.index

    if not exists(f'./data/history/{name}_hvs.csv'):
        return None, None, None
    with open(f'./data/history/{name}_hvs.csv', 'r') as f:
        df_hvs = pd.read_csv(f, index_col=[0,1], names=['HV','Profile']+list(range(2000)), header=None)
        df_hvs = df_hvs.T

    if not exists(f'./data/history/{name}_info_hvs.csv'):
        return None, None, None
    with open(f'./data/history/{name}_info_hvs.csv', 'r') as f:  
        columns = ['id'] + TRAITS + ['birth', 'age']
        df_info = pd.read_csv(f, index_col=0, names=columns, header=None)
        df_info.head()

    return df_group, df_hvs, df_info

def get_dfs(df, ihv):
    dfs = df[ihv]
    dfs = dfs[(dfs['reward'].notna())]
    cols = dfs.columns.drop('action_name')
    dfs[cols] = dfs[cols].apply(pd.to_numeric)    
    #dfs['target'] = pd.to_numeric(dfs['target'], downcast='unsigned')
    dfs['target'] = dfs['target'].astype('Int64')
    dfs['pow_o_res'] = (dfs['power']-dfs['resistance'])/((dfs['power']+dfs['resistance']))
    dfs['day'] = dfs.index
    # dfs.info()
    return dfs


def load_hvs(name):            
    if not exists(f'./data/history/{name}_hvs.csv'):
        return None
    with open(f'./data/history/{name}_hvs.csv', 'r') as f:
        # columns = ["id"] + list(range(200))
        # df = pd.read_csv(f, names=columns, index_col="id")
        df = pd.read_csv(f)
    return df

