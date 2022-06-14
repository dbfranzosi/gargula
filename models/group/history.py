from settings import *
from collections import namedtuple, deque

Indicators = namedtuple('Indicators',
                        ('genes', 'traits', 'actions'))

class GroupHistory:

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
        
        # This is very inneficient, update instead!        
        indicators = Indicators(*zip(*indicators))        
        length = indicators.__len__()        

        # print('ind=',indicators)       
        # print('ind_dict=', indicators._asdict())
        # for name, value in indicators._asdict().items():
        #     print('name=', name)
        #     print('valeu=', value)
        
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

    def get_actions_avg(self):        
        actions = [hv.action.name for hv in self.owner.hvs.values()]        
        actions_avg = [actions.count(action) for action in ACTIONS]
        return actions_avg

    def update(self):
        self.push(self.get_genes(), self.get_traits_avg(), self.get_actions_avg())        

    