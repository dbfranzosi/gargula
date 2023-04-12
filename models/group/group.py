from settings import *
import numpy as np
from models.hv.hv import Hv
from reality.biology import biology
from reality.geography import eden
import random
from .history import *
import pandas as pd
import pickle
import time
import glob, os

class Group:
    '''
    Class to define a group of homo-virtualis.
    '''
    
    def __init__(self, id=0, name='', home=None, hvs={}):
        self.id = id
        self.id_last = 0 # id of last hv +1
        self.name = name
        self.home = home
        if home:
            home.groups[id] = self
        self.hvs = hvs
        self.biology = biology
        self.history = GroupHistory(self, capacity=100)        
    
    def nr_hvs(self):
        return len(self.hvs)  
    
    def get_list_ids(self):
        return list(self.hvs.keys())

    def extinction(self):
        if (self.nr_hvs() <= 0):
            print('A group was extinct.')
            print(self.get_info())
            del self.home.groups[self.id]

    def check_deaths(self):
        hv_keys = list(self.hvs.keys()) # Check deaths
        for ihv in hv_keys: 
            hv = self.hvs[ihv]
            hv.death()   

    def interact(self):
        hv_keys = list(self.hvs.keys()) # fix to avoid change in the loop        
        random.shuffle(hv_keys) # random initiative            
        for ihv in hv_keys:             
            hv = self.hvs[ihv]            
            hv.act()
            hv.aging()
            hv.history.update()

    def generate_gargalo(self, nr=10):
        lst_names = ['Adam', 'Eva', 'Lilith', 'Caim', 'Abel', 'Raul', 'Che', 'Karl', 'Lenin', 'José']
        hvs = {}  # these are the first homovirtualis that pass the bottleneck that defined the species.       
        rng = np.random.default_rng(RANDOM_SEED)
        for id in range(nr):   
            name = lst_names[id] if id<len(lst_names) else f'HV_{id}' 
            haploid_father = rng.integers(0, 2, size=GEN_SIZE)
            haploid_mother = rng.integers(0, 2, size=GEN_SIZE)
            hv = Hv(group=self, name=name, haploid_father=haploid_father, haploid_mother=haploid_mother)            
            # make id=0 imortal for test purpose
            # if (id == 0):
            #     hv.genes.phenotype.traits['energy_pool'] = 100000.
            #     hv.energy = 100000.

    def get_info(self):
        str_hvs = ', '.join([hv.name for hv in self.hvs.values()])        
        return f'Group {self.name} in area {self.home.name} has {self.nr_hvs()} homo-virtualis: {str_hvs}'
    
    def get_features(self):        
        return pd.DataFrame.from_dict({hv.id : hv.visible.features for hv in self.hvs.values()}, orient='index', columns=FEATURES)
    
    def get_traits(self):
        traits_dict = dict.fromkeys(TRAITS)        
        k, v = self.hvs.keys(), self.hvs.values()        
        for trait in TRAITS:
            traits_dict[trait] = [hv.genes.phenotype.traits[trait] for hv in v]
        df = pd.DataFrame.from_dict(traits_dict)
        df['id'] = k        
        df = df.set_index('id') 
        return df

    def get_actions(self):        
        return pd.DataFrame.from_dict({hv.id : [hv.action.name, hv.action.description] for hv in self.hvs.values()}, 
                        orient='index', columns=['action', 'description'])
    
    def get_profiles(self):                
        #profile = pd.DataFrame.from_dict({hv.id : hv.visible.features for hv in self.hvs.values()}, orient='index', columns=FEATURES)        
        profile = self.get_features()
        to_drop = [trait for trait in TRAITS if trait in FEATURES]
        profile.drop(to_drop, axis=1, inplace=True)     
        
        traits = self.get_traits()
        actions = self.get_actions()

        extra = pd.DataFrame.from_dict({hv.id : [hv.age] for hv in self.hvs.values()}, orient='index', columns=['age'])
               
        profile = pd.concat([extra, profile, traits, actions], axis=1)    
        return profile

    def get_family(self):        
        family = []
        hvids = self.hvs.keys()
        generations = [hv.generation for hv in self.hvs.values()]
        min_generation = min(generations)
        set_generations = set(generations)
        i = dict.fromkeys(set_generations, 0)
        for hvid, hv in self.hvs.items():  
            generation = hv.generation 
            x_max = generations.count(generation)  
            y_max = len(set_generations)
            str_url = f'assets/hv{hvid}.svg'
            if not os.path.exists(str_url):
                print(str_url, " doesn't exist.")
                str_url = 'assets/gargula-inverted.jpg'                
            family.append({'data': {'id': hvid, 'label': hv.name, 'url': str_url},
                            'position': {'x': i[generation]*600./x_max, 'y': (hv.generation - min_generation) * 500./y_max}})
            i[generation] += 1
            #print(family)
            if hv.parents is None:
                pass
            else:          
                if (hv.parents[0] in hvids):      
                    family.append({'data': {'source': hvid, 'target': hv.parents[0], 'role': 'father'}})
                if (hv.parents[1] in hvids):      
                    family.append({'data': {'source': hvid, 'target': hv.parents[1], 'role': 'mother'}})          
        return family
    
    def add_hv(self, hv):                       
        hv.group = self
        hv.id = self.id_last
        self.hvs[hv.id] = hv
        self.id_last += 1 

    def save(self):
        filename = f'./data/groups/{self.name}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        filename = f'./data/groups/{name}.pickle'
        with open(filename, 'rb') as f:
            self = pickle.load(f)     
        return self
    
    def merge(self, name):
        filename = f'./data/groups/{name}.pickle'
        with open(filename, 'rb') as f:
            group_add = pickle.load(f)             
            if (self.biology == group_add.biology):
                for hv in group_add.hvs.values():
                    self.add_hv(hv)
                    hv.name = name + hv.name                    

                return self, True
            else:
                return self, False

    def clean(self):
        # remove avatars
        for f in glob.glob("assets/hv*.svg"):
            os.remove(f)
        self = Group(name='Gargalo', home=eden)
        str_tmp = f'./data/history/tmp.csv'
        if os.path.exists(str_tmp):
            os.remove(str_tmp)
        str_tmp = f'./data/history/tmp_hvs.csv'
        if os.path.exists(str_tmp):
            os.remove(str_tmp)        
        return self

    def write_histories(self):

        with open(f'./data/history/{self.name}.csv', 'a') as fout:
            str_tmp = f'./data/history/tmp.csv'
            if os.path.exists(str_tmp):
                fin = open(str_tmp, "r")
                data = fin.read()
                fin.close()                
                fout.write(data)
                os.remove(str_tmp)

        with open(f'./data/history/{self.name}_hvs.csv', 'a') as fout:
            str_tmp = f'./data/history/tmp_hvs.csv'
            if os.path.exists(str_tmp):
                fin = open(str_tmp, "r")
                data = fin.read()
                fin.close()                
                fout.write(data)
                os.remove(str_tmp)

        with open(f'./data/history/{self.name}_info_hvs.csv', 'a') as fout:
            str_tmp = f'./data/history/tmp_info_hvs.csv'
            if os.path.exists(str_tmp):
                fin = open(str_tmp, "r")
                data = fin.read()
                fin.close()                
                fout.write(data)
                os.remove(str_tmp)

    def load_history(self):
        with open(f'./data/history/{self.name}.csv', 'r') as fin:
            df = pd.read_csv(fin, header=None)
            
        with open(f'./data/history/{self.name}_hvs.csv', 'r') as fin:
            df_hvs = pd.read_csv(fin, index_col=[0,1], names=['HV','Profile']+list(range(2000)), header=None)
            df_hvs = df.T
        
        with open(f'./data/history/{self.name}_info_hvs.csv', 'r') as fin:
            columns = TRAITS + ['birth', 'age']
            df_info_hvs = pd.read_csv(fin, index_col=0, names=columns, header=None)

        return df, df_hvs, df_info_hvs




gargalo = Group(name='Gargalo', home=eden)
# gargalo.generate_gargalo()
# conception = Group(name='Conception')



    
