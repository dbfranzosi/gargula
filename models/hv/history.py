from settings import *
from collections import namedtuple, deque
import pandas as pd

Indicators = namedtuple('Indicators',
                        ('action_counter', 'action_name', 'reward', 'power', 'resistance'))
# remove action_counter eventually

class HvHistory:

    def __init__(self, owner, capacity=500):
        self.memory = deque([], maxlen=capacity)
        self.owner = owner
        self.capacity = capacity        

    def push(self, *args):
        """Save a transition"""        
        self.memory.append(Indicators(*args))

    def __len__(self):
        return len(self.memory)

    def count_actions(self):           
        if (self.__len__() == 0):
            action_counter = {ACTIONS[i] : 0 for i in range(len(ACTIONS))}
        else:
            #action_counter = self.memory[-1].action_counter
            action_counter = {ACTIONS[i] : self.memory[-1].action_counter[i] for i in range(len(ACTIONS))}

        #print('action_counter dict=', action_counter)        
        action_counter[self.owner.action.name] += 1        
        action_counter = np.array([action_counter[action] for action in ACTIONS])       
        #print('action_counter=', action_counter)        
        return action_counter

    def get_action(self):
        pass

    def update(self):
        if (hasattr(self.owner.action, 'power') and hasattr(self.owner.action, 'resistance')):
            power = self.owner.action.power
            resistance = self.owner.action.resistance
        else:
            power = None
            resistance = None       

        self.push(self.count_actions(), self.owner.action.name, self.owner.action.reward, power, resistance)

    def get_indicators(self):

        if (self.__len__() == 0):
            y_actions = {action : 0.0 for action in ACTIONS}
            return y_actions, [], [], [], []
        else:
            indicators = self.memory            
            indicators = Indicators(*zip(*indicators))                
            
            val = np.array(indicators.action_counter)            
            val = np.transpose(val)                     
            y_actions = {ACTIONS[i] : val[i] for i in range(len(ACTIONS))}         
            
            val = np.array(indicators.action_name)            
            val = np.transpose(val)                     
            y_action_name = val
            
            val = np.array(indicators.reward)            
            val = np.transpose(val)                     
            y_reward = val

            val = np.array(indicators.power)            
            val = np.transpose(val)                     
            y_power = val

            val = np.array(indicators.resistance)            
            val = np.transpose(val)                     
            y_resistance = val

            return y_actions, y_action_name, y_reward, y_power, y_resistance

    def to_df(self):
        y_actions, y_action_name, y_reward, y_power, y_resistance = self.get_indicators()        
        #df = pd.DataFrame({i : [y_action_name[i], y_reward[i], y_power[i], y_resistance[i]] for i in range(len(y_reward))}, index = [self.owner.id])
        df = pd.DataFrame({i : [y_reward[i]] for i in range(len(y_reward))}, index = [self.owner.id])        
        return df

    def save(self, header=False): 
        df = self.to_df()

        with open(f'./data/history/{self.owner.group.name}_hvs.csv', 'a') as f:
            df.to_csv(f, mode='a', header=header)


