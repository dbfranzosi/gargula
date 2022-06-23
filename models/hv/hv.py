import numpy as np
from .genes import Genes
from .mind.mind import *
from .visible import Visible
from .actions import Rest
from settings import *
from .history import HvHistory

id_last = 0

class Hv:
        
    def __init__(self, group=None, haploid_father=None, haploid_mother=None, energy = 0.0, generation=1, name=''):

        global id_last
        self.id = id_last
        id_last += 1
        self.group = group
        group.hvs[self.id] = self          
        self.genes = Genes(haploid_father,haploid_mother)         
        self.pregnant=0 
        self.age=0    
        self.generation = generation             
        if energy==0.0:
            self.energy = self.genes.phenotype.traits['energy_pool']
        else:
            self.energy = energy
        self.name = name
        self.parents = None
        self.visible = Visible(self)                
        self.action = Rest(self) 

        #self.mind = PonderMind(self, group)         
        #self.mind = MemoryGraphMind(self, 1000)
        #self.mind = GraphMind(self, 1000)
        self.mind = GraphDQLMind(self, 1000)

        self.history = HvHistory(self, 200)

    def act(self):
        self.action = self.mind.decide_action()        
        self.action.effects()        
        self.mind.update_memory()  
        self.mind.optimize_mind()           

    def aging(self):
        self.age += 1
        self.visible.update(self)
        if self.age > AGE_BABY:
            pass
            #self.mind = MemoryGraphMind(self, 1000)   

        if self.pregnant >= AGE_BABY:
            self.pregnant = 0
        if self.pregnant>0:
            self.pregnant += 1            
    
    def death(self):
        if (self.energy<0):
            #print('=== Someone Died.')
            #self.visualize()
            del self.group.hvs[self.id]                                    

    def make_baby(self, mother):
        haploid_father = self.genes.meiosis()
        haploid_mother = mother.genes.meiosis()
        #generation = max(self.generation, mother.generation) + 1
        generation = max(self.generation, mother.generation) + 1 
        baby_name = f'{mother.name.split()[0]} {generation}'
        baby = Hv(group=self.group, name=baby_name, haploid_father=haploid_father, 
                haploid_mother=haploid_mother, energy=5*UNIT_ENERGY, generation=generation) 
        baby.parents = [self.id, mother.id]
        #baby.mind = MemoryGraphMind(baby, 1000)
        mother.pregnant += 1

        #print('=== A baby was born!')
        #self.visualize(show_genes=True)
        #mother.visualize(show_genes=True)
        #baby.visualize(show_genes=True)

        return baby

    def visualize(self, show_genes=False, show_action=False, show_memory = False, show_visible=False):
        str = f'{self.name}({self.id}): age={self.age}, energy={self.energy}, pregnant={self.pregnant}. '
        if show_action:
            str += f'{self.action.description}'
        if show_genes:
            str += f'\n{self.genes.sequence[0]}\n'
            str += f'{self.genes.sequence[1]}'
        if show_visible:
            str += f'\n{self.visible.features}\n'    
        print(str)
        if show_memory:
            print('memory=', self.mind.memory.__len__())
        
    def get_genes(self):
        return self.genes.sequence[0]+self.genes.sequence[1]

        

    
