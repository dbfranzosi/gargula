from settings import *
from collections import namedtuple, deque
import pandas as pd

Indicators = namedtuple('Indicators',
                        ('action_name', 'reward', 'target', 'power', 'resistance'), defaults=(None, None, None, None, None))
Counters = namedtuple('Counters',
                        ('action_counter'))


class HvHistory:

    def __init__(self, owner, capacity=500):
        self.memory = deque([], maxlen=capacity)
        self.counter = deque([], maxlen=capacity)
        self.owner = owner
        self.capacity = capacity        

    def push(self, *args):
        """Save indicators"""        
        self.memory.append(Indicators(*args))

    def push_counter(self, *args):
        """Save counter"""                
        self.counter.append(Counters(*args))

    def __len__(self):
        return len(self.memory)
    def __len_counter__(self):
        return len(self.counter)

    def count_actions(self):           
        if (self.__len_counter__() == 0):
            action_counter = {ACTIONS[i] : 0 for i in range(len(ACTIONS))}
        else:
            #action_counter = self.memory[-1].action_counter
            action_counter = {ACTIONS[i] : self.counter[-1].action_counter[i] for i in range(len(ACTIONS))}

        #print('action_counter dict=', action_counter)        
        action_counter[self.owner.action.name] += 1        
        action_counter = np.array([action_counter[action] for action in ACTIONS])       
        #print('action_counter=', action_counter)        
        return action_counter

    def update(self):
        if (hasattr(self.owner.action, 'power') and hasattr(self.owner.action, 'resistance')):
            power = self.owner.action.power
            resistance = self.owner.action.resistance
            target = self.owner.action.target.id
        else:
            power = None
            resistance = None    
            target = None   

        self.push(self.owner.action.name, self.owner.action.reward, target, power, resistance)
        self.push_counter(self.count_actions())

    def get_indicators(self):
        if (self.__len__() == 0):
            indicators = Indicators()
        else:
            indicators = self.memory            
            indicators = Indicators(*zip(*indicators))     

        return indicators

    def get_counter(self):

        if (self.__len__() == 0):
            y_actions = {action : 0.0 for action in ACTIONS}
            return y_actions
        else:
            counter = self.counter 
            counter = Counters(*zip(*counter))     
            
            val = np.array(counter.action_counter)            
            val = np.transpose(val)                     
            y_actions = {ACTIONS[i] : val[i] for i in range(len(ACTIONS))}         

            return y_actions

    def to_df(self):
        indicators = self.get_indicators()        
        # df = pd.DataFrame({i : [y_action_name[i], y_reward[i], y_power[i], y_resistance[i]] for i in range(len(y_reward))}, index = [self.owner.id])
        # df = pd.DataFrame({i : [y_reward[i]] for i in range(len(y_reward))}, index = [self.owner.id])        
        iterables = [[self.owner.id], indicators._asdict().keys()]                
        index = pd.MultiIndex.from_product(iterables, names=["HV", "Profile"])        
        columns = list(range(max(1,self.__len__())))                
        data = np.array(list(indicators._asdict().values()))        
        df = pd.DataFrame(data, index, columns)        
        
        return df

    def save(self, header=False): 
        df = self.to_df()

        # with open(f'./data/history/{self.owner.group.name}_hvs.csv', 'a') as f:
        with open(f'./data/history/tmp_hvs.csv', 'a') as f:
            df.to_csv(f, mode='a', header=header)


