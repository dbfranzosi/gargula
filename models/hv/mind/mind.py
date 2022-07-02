import numpy as np
import math
import random
import networkx as nx
from ..actions import *
from ..visible import *
from settings import *
#from .interaction_network import InteractionNetwork
from .memory import *
from .interaction_network import *

class BaseMind:
    ''' Basic class for mind. Only rest.'''
    def __init__(self, owner):          
        self.owner = owner 
        self.group = owner.group  
        self.nr_active_actions = self.group.nr_hvs()*nr_targeted_actions + nr_nontarg_actions
        self.memory = ReplayMemory()
        
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

# Training
#BATCH_SIZE = 128
#BATCH_SIZE = 20
BATCH_SIZE = 5
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class GraphMind(MemoryGraphMind):
    def __init__(self, owner, memory_capacity):
        super().__init__(owner, memory_capacity)
        self.policy_net = InteractionNetwork(n_objects, object_dim, n_relations, effect_dim, nr_class_actions).to(device)
        self.target_net = InteractionNetwork(n_objects, object_dim, n_relations, effect_dim, nr_class_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())        

    def decide_action(self):                
        self.state = torch.tensor(perception(self.group), device=device, dtype=torch.float)         
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.owner.age / EPS_DECAY)        
        if sample > eps_threshold:            
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the largest expected reward.                
                RS = sender_relations.expand(1, -1 , -1)           
                RR = receiver_relations.expand(1, -1 , -1) 
                X = self.state.expand(1, -1, -1)
                predicted = self.policy_net(X, RS, RR)     
                # put in batch form with 1 entry                           
                self.action = predicted.argmax().expand(1)                 
                target_nr, action_nr = divmod(int(self.action), nr_class_actions) 
                action_code = [ACTIONS[action_nr], target_nr]                
                action = get_action(action_code, self.owner)   
        else:
            nr_hvs = self.group.nr_hvs()
            target_nr, action_nr = random.randrange(0, nr_hvs), random.randrange(0, nr_class_actions)
            self.action = torch.tensor([target_nr*nr_class_actions + action_nr], device=device, dtype=torch.int64)
            action_code = [ACTIONS[action_nr], target_nr]            
            action = get_action(action_code, self.owner)    
        self.reward = torch.tensor([action.reward], device=device, dtype=torch.float) 
        return action

class GraphDQLMind(GraphMind):
    ''' Mind based on DQL (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        with a state defined by the interaction network.'''
    def optimize_mind(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Repeat RS, RR to convert (Np, Npp) into (n_batch, Np, Npp)
        RS = sender_relations.expand(BATCH_SIZE, -1 , -1)           
        RR = receiver_relations.expand(BATCH_SIZE, -1 , -1)                 

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        # print('sb=', state_batch.shape)
        # print('actb=', action_batch)
        
        predicted = self.policy_net(state_batch, RS, RR)        

        # print('pred=', predicted)
        # print('pview=', predicted.view(BATCH_SIZE, -1))
        # print(action_batch)

        # Flatten O[Do, Np] to pick the action (one-index)
        state_action_values = predicted.view(BATCH_SIZE, -1).gather(1, action_batch)
        
        # print('sact=', state_action_values)
        # print('===============')
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions are computed based
        # on the "older" target_net; selecting their best reward.
        
        predicted_target = self.target_net(next_state_batch, RS, RR)        
        # Get the 1-index value (that maps to action) for each event in batch
        next_state_values = predicted_target.view(BATCH_SIZE, -1).max(dim=1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch   

        # print(state_batch.size(), ' ', RS.size(), ' ', RR.size())
        # print('predicted=', predicted)
        # print('state=', state_action_values)
        # print('expected=', expected_state_action_values)
        # print('reward=', reward_batch)
        # # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        #print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if self.owner.age % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

