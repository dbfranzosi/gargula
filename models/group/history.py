from settings import *
from collections import namedtuple, deque
import pandas as pd
from os.path import exists
#import h5py

Indicators = namedtuple('Indicators',
                        ('genes', 'traits', 'actions'))

class GroupHistory:

    def __init__(self, owner, capacity=100):
        self.memory = deque([], maxlen=capacity)
        self.owner = owner
        self.capacity = capacity

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
        # print('gens=', [hv.genes.sequence[0]+hv.genes.sequence[1] for hv in self.owner.hvs.values()])
        # print('sum gen=', np.sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in self.owner.hvs.values()], axis=0)/(2*self.owner.nr_hvs())) 
        return np.sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in self.owner.hvs.values()], axis=0)/(2*self.owner.nr_hvs())
    
    def get_traits_avg(self):        
        traits_avg = np.zeros(len(TRAITS))
        traits_avg = np.array([np.mean([hv.genes.phenotype.traits[trait] for hv in self.owner.hvs.values()]) for trait in TRAITS])
        return traits_avg

    def get_actions_count(self):        
        actions = [hv.action.name for hv in self.owner.hvs.values()]        
        actions_count = np.array([actions.count(action) for action in ACTIONS])
        return actions_count

    def update(self):
        self.push(self.get_genes(), self.get_traits_avg(), self.get_actions_count())   

    def to_df(self):
        y_gen, y_traits, y_actions = self.get_indicators()        
        df = pd.concat([pd.DataFrame(y_gen), pd.DataFrame(y_traits), pd.DataFrame(y_actions)], axis=1)
        return df

    def save(self, header=False): 
        df = self.to_df()

        # with open(f'./data/history/{self.owner.name}.csv', 'a') as f:
        with open(f'./data/history/tmp.csv', 'a') as f:
            df.to_csv(f, mode='a', index=False, header=header)


        

    