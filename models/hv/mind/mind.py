import numpy as np
import math
import random
import networkx as nx
from ..actions import *
from ..visible import *
from settings import *
#from .interaction_network import InteractionNetwork
from .gdql import *
from .interaction_network import *

class BaseMind:
    ''' Basic class for mind. Only rest.'''
    def __init__(self, owner):          
        self.owner = owner 
        self.group = owner.group  
        self.nr_active_actions = self.group.nr_hvs()*nr_targeted_actions + nr_nontarg_actions
        self.memory = ReplayMemory(100)
        
    def decide_action(self, action_nr=nr_actions-1):        
        action_code = lst_actions[action_nr]        
        action = get_action(action_code, self.owner)  
        return action    

    def decide_passive(self, action_name, active_hv):
        return True

    def update_memory(self):
        pass

    def optimize_mind(self):
        pass

class RandomMind(BaseMind):

    def decide_action(self):        
        return super().decide_action(action_nr = np.random.randint(0, self.nr_active_actions))

    def decide_passive(self, action_name, active_hv):        
        return np.random.randint(0, 2) 
        
class PonderMind(BaseMind):
    ''' Ponder can be e.g. the output of the DQN.'''
    def __init__(self, owner):
        super().__init__(owner)
        a = np.zeros(nr_actions)
        #targeted actions        
        nr = self.group.nr_hvs()*nr_targeted_actions
        a[0:nr] = np.random.random(size = nr)         
        # non-targeted actions        
        a[-nr_nontarg_actions:] = np.random.random(size = nr_nontarg_actions)         
        self.ponder = a
    
    def decide_action(self):
        action_nr = np.argmax(self.ponder) # decide on the largest weight
        return super().decide_action(action_nr = action_nr) 
        
    def decide_passive(self, action_name, active_hv):
        b = np.random.random(size=nr_passive_actions)
        ponder = b
        return  np.rint(ponder[lst_passive_actions[action_name]])

class MemoryGraphMind(PonderMind):
    def __init__(self, owner, memory_capacity):
        super().__init__(owner)
        self.memory = ReplayMemory(memory_capacity)
        self.state = torch.tensor(perception(self.group), device=device) 
        self.next_state = None
        self.reward = None
        # action representation in mind is the index of predict tensor
        self.action = torch.tensor([-1], device=device, dtype=torch.float) 

    def decide_action(self):                
        action = super().decide_action()
        self.state = torch.tensor(perception(self.group), device=device, dtype=torch.float) 
        self.reward = torch.tensor([action.reward], device=device, dtype=torch.float)   
        return action     

    def update_memory(self):        
        self.next_state = torch.tensor(perception(self.group), device=device, dtype=torch.float) 
        self.memory.push(self.state, self.action, self.next_state, self.reward)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class GraphMind(MemoryGraphMind):
    def __init__(self, owner, memory_capacity):
        super().__init__(owner, memory_capacity)
        self.policy_net = InteractionNetwork(n_objects, object_dim, n_relations, effect_dim, n_actions).to(device)
        self.target_net = InteractionNetwork(n_objects, object_dim, n_relations, effect_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())        

    def decide_action(self):        
        ### MESS HERE
        self.state = torch.tensor(perception(self.group), device=device, dtype=torch.float)         
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.owner.age / EPS_DECAY)        
        if sample > eps_threshold:
            #print('exploitation')
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #self.policy_net(self.state).max(1)[1].view(1, 1)                
                #data = np.array([self.state])
                #objects, sender_relations, receiver_relations = get_batch(data, 1) 
                RS = sender_relations.expand(1, -1 , -1)           
                RR = receiver_relations.expand(1, -1 , -1) 
                X = self.state.expand(1, -1, -1)
                predicted = self.policy_net(X, RS, RR)
                #print(predicted)
                self.action = predicted.argmax().expand(1)
                print('action=', self.action)
                target_nr, action_nr = divmod(int(self.action), nr_class_actions) 
                action_code = [lst_class_actions[action_nr], target_nr]
                #print(action_code)
                action = get_action(action_code, self.owner)                  
        else:
            #print('exploration')
            nr_hvs = self.group.nr_hvs()
            target_nr, action_nr = random.randrange(0, nr_hvs), random.randrange(0, nr_class_actions)
            self.action = torch.tensor([target_nr*nr_hvs + action_nr], device=device, dtype=torch.int64)
            action_code = [lst_class_actions[action_nr], target_nr]
            #print(action_code)
            action = get_action(action_code, self.owner)  
            #print(torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long))
        self.reward = torch.tensor([action.reward], device=device, dtype=torch.float) 
        return action
        
        

class GraphDQLMind(GraphMind):
    def optimize_mind(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        print('training')
        print(batch.action)
        print(batch.state)
        
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        RS = sender_relations.expand(BATCH_SIZE, -1 , -1)           
        RR = receiver_relations.expand(BATCH_SIZE, -1 , -1)                 

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        print(state_batch.size())
        print(RS.size())
        print(RR.size())
        predicted = self.policy_net(state_batch, RS, RR)
        state_action_values = predicted.gather(1, action_batch)
        
        print('state_action_values=', state_action_values)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.       
        # next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # ===> PICK RIGHT DIMENSION HERE 
        predicted_target = self.target_net(next_state_batch, RS, RR)        
        next_state_values = predicted_target.max(1)[0].detach()
        print('next_state_values=', next_state_values)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch        

        # # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # # Optimize the model
        # optimizer.zero_grad()
        # loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # optimizer.step()

        # # Update the target network, copying all weights and biases in DQN
        # if self.owner.age % TARGET_UPDATE == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())




# class GraphMind(PonderMind):
    
#     n_objects  =  MAX_HVS_IN_GROUP # number of hvs (nodes)
#     object_dim = NR_FEATURES # features in visible
#     n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
#     effect_dim = 100 #effect's vector size
#     output_dim = nr_class_actions

#     def __init__(self, owner, group):
#         super().__init__(owner, group)
#         target_net = InteractionNetwork(n_objects, object_dim, n_relations, effect_dim)
#         target_net.eval()
#         #predict = 
#         print()

def perception(group):    
    nr_hvs = group.nr_hvs()
    obj = np.zeros((MAX_HVS_IN_GROUP, NR_FEATURES))    
    obj[0:nr_hvs] = np.array([hv.visible.features for hv in group.hvs.values()])
    return obj
        
        




