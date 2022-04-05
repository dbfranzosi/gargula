from settings import *
import numpy as np
from models.hv.hv import Hv
from reality.geography import eden
import random


class Group:
    
    def __init__(self, id=0, name='', home=None, hvs={}):
        self.id = id
        self.name = name
        self.home = home
        if home:
            home.groups[id] = self
        self.hvs = hvs

    def nr_hvs(self):
        return len(self.hvs)  

    def extinction(self):
        if (self.nr_hvs() <= 0):
            print('A group was extinct.')
            self.visualize()
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
            hv.visualize(show_action=True, show_memory=True)          

    def generate_gargalo(self, nr=10):
        lst_names = ['Adam', 'Eva', 'Lilith', 'Caim', 'Abel', 'Raul', 'Che', 'Karl', 'Lenin', 'José']
        hvs = {}  # these are the first homovirtualis that pass the bottleneck that defined the species.       
        rng = np.random.default_rng(RANDOM_SEED)
        for id in range(nr):   
            name = lst_names[id] if id<len(lst_names) else f'HV_{id}' 
            haploid_father = rng.integers(0, 2, size=GEN_SIZE)
            haploid_mother = rng.integers(0, 2, size=GEN_SIZE)
            hv = Hv(group=self, name=name, haploid_father=haploid_father, haploid_mother=haploid_mother)
            hv.age = 50 # all adults 

    def visualize(self):
        str_hvs = ', '.join([hv.name for hv in self.hvs.values()])
        #print(self.hvs.values().name)
        print(f'Group {self.name} in area {self.home.name} has {self.nr_hvs()} homo-virtualis: {str_hvs}')

    # Graphical tools
    def plot_genes_init(self, ax):
        #x = np.arange(GEN_SIZE)
        y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in self.hvs.values()])/(2*self.nr_hvs())
        steps = ax.stairs(y, fill=True)  
        ax.set_ylim(0.,1.2)
        return steps      

    def plot_genes_update(self, ax, steps):        
        y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in self.hvs.values()])/(2*self.nr_hvs())
        steps.set_data(y)

gargalo = Group(name='Gargalo', home=eden)
gargalo.generate_gargalo()
conception = Group(name='Conception')
