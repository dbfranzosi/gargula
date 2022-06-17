from settings import *
from collections import namedtuple, deque

Indicators = namedtuple('Indicators',
                        ('genes', 'traits', 'actions'))

class HvHistory:

    def __init__(self, owner, capacity):
        self.memory = deque([], maxlen=capacity)
        self.owner = owner

    def push(self, *args):
        """Save a transition"""        
        self.memory.append(Indicators(*args))

    def __len__(self):
        return len(self.memory)

    def get_indicators(self):
        indicators = self.memory             
                
        indicators = Indicators(*zip(*indicators))        
        length = indicators.__len__()        
        
        val = np.array(indicators.genes)        
        val = np.transpose(val)         
        y_gen = {gen : val[gen] for gen in range(GEN_SIZE)}

        val = np.array(indicators.traits)        
        val = np.transpose(val)             
        y_traits = {TRAITS[i] : val[i] for i in range(len(TRAITS))}                        

        val = np.array(indicators.actions)        
        val = np.transpose(val)             
        y_actions = {ACTIONS[i] : val[i] for i in range(len(ACTIONS))}                        
        
        return y_gen, y_traits, y_actions
    
    def get_genes(self):               
        return self.owner.genes.sequence[0]+self.owner.genes.sequence[1]
    
    def get_traits(self):                
        traits = [self.owner.genes.phenotype.traits[trait] for trait in TRAITS]
        return traits

    def get_actions(self):                
        return [self.owner.action.name]

    def update(self):
        self.push(self.get_genes(), self.get_traits(), self.get_actions())   

