from settings import *
from collections import namedtuple, deque

Indicators = namedtuple('Indicators',
                        ('actions', 'passive_actions'))

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
        #print('indicators=', indicators)         
        indicators = Indicators(*zip(*indicators))        
        length = indicators.__len__()        

        # print('ind=',indicators)       
        # print('ind_dict=', indicators._asdict())
        # for name, value in indicators._asdict().items():
        #     print('name=', name)
        #     print('value=', value)
        
        val = np.array(indicators.actions)            
        val = np.transpose(val)             
        # print(val)
        y_actions = {ACTIONS[i] : val[i] for i in range(len(ACTIONS))} 
        #print(y_actions)
        
        return y_actions    

    def count_actions(self):           
        if (self.__len__() == 0):
            action_counter = {ACTIONS[i] : 0 for i in range(len(ACTIONS))}
        else:
            #action_counter = self.memory[-1].actions
            action_counter = {ACTIONS[i] : self.memory[-1].actions[i] for i in range(len(ACTIONS))}

        #print('action_counter dict=', action_counter)        
        action_counter[self.owner.action.name] += 1        
        action_counter = np.array([action_counter[action] for action in ACTIONS])       
        #print('action_counter=', action_counter)        
        return action_counter

    def update(self):
        self.push(self.count_actions(), None)   

