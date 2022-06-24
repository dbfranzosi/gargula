import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from ..actions import nr_class_actions
from settings import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.autograd.set_detect_anomaly(True)

n_objects  =  MAX_HVS_IN_GROUP # number of hvs (nodes)
object_dim = NR_FEATURES # features in visible
n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
effect_dim = 100 #effect's vector size
output_dim = nr_class_actions

receiver_relations = np.zeros((n_objects, n_relations))
sender_relations   = np.zeros((n_objects, n_relations))

# R_S R_R are fixed and could be computed only once...
cnt = 0
for i in range(n_objects):
    for j in range(n_objects):
        if(i != j):
            receiver_relations[i, cnt] = 1.0
            sender_relations[j, cnt]   = 1.0
            cnt += 1

sender_relations   = torch.tensor(sender_relations, device=device, dtype=torch.float)
receiver_relations = torch.tensor(receiver_relations, device=device, dtype=torch.float)

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)        
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x

class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ObjectModel, self).__init__()

        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size), #rest, eat (non-targeted). sex, attack (targeted)
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size, n_objects, output_size] out ~ rest, eat (non-targeted). sex, attack (targeted)
        '''

        batch_size, n_objects, input_size = x.size()
        x = x.view(-1, input_size)        
        x = self.layers(x)
        x = x.view(batch_size, n_objects, self.output_size)
        return x

class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, object_dim, n_relations, effect_dim, output_dim):
        super(InteractionNetwork, self).__init__()
        
        self.relational_model = RelationalModel(2*object_dim, effect_dim, 150)
        self.object_model     = ObjectModel(object_dim + effect_dim, 100, output_dim)
    
    def forward(self, objects, sender_relations, receiver_relations):
        senders   = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers], 2))
        effect_receivers = receiver_relations.bmm(effects)
        predicted = self.object_model(torch.cat([objects, effect_receivers], 2))
        # print('objects=', objects.size())
        # print('senders=', senders.size())
        # print('receivers=',receivers.size())
        # print('effects=',effects.size())
        # print('effect_receivers=',effect_receivers.size())
        # print('predicted=',predicted.size())
        return predicted

