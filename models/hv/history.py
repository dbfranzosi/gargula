from settings import *
from collections import namedtuple, deque

Indicators = namedtuple('Indicators',
                        ('action_counter'))

class HvHistory:

    def __init__(self, owner, capacity):
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
        print(indicators)
        indicators = Indicators(*zip(*indicators))        
        length = indicators.__len__()        

        print('ind=',indicators)       
        print('ind_dict=', indicators._asdict())
        for name, value in indicators._asdict().items():
            print('name=', name)
            print('value=', value)
        
        val = np.array(indicators.action_counter)            
        val = np.transpose(val)             
        print(val)
        y_actions = {ACTIONS[i] : val[i] for i in range(len(ACTIONS))}                        
        
        return y_actions    

    def count_actions(self):   
        if (self.__len__() == 0):
            action_counter = {ACTIONS[i] : 0 for i in range(len(ACTIONS))}
        else:
            action_counter = self.memory[-1]
        print(action_counter)

        #y_actions[self.owner.action.name] += 1        
        #return y_actions

    def update(self):
        self.push(self.count_actions())   

