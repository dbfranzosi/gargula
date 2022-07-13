import random
from collections import namedtuple, deque
from settings import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity=100):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""        
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)    

    def __len__(self):
        return len(self.memory)

def perception(group):    
    nr_hvs = group.nr_hvs()
    obj = np.zeros((MAX_HVS_IN_GROUP, NR_FEATURES))    
    obj[0:nr_hvs] = np.array([hv.visible.features for hv in group.hvs.values()])
    return obj


