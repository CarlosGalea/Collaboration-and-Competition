#Import Packages
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """
    Helper function for initializing the hidden layer weights with random noise, preventing gradients from exploding or vanishing
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        
        """
        Initialize parameters and build model
        
        state_size: Environment's state size
        action_size: Agents' action size
        seed: Random seed
        """
        self.seed = torch.manual_seed(seed)
        
        #1 batch normalization layer
        self.bn1 = nn.BatchNorm1d(state_size)
        
        #Initializing 3 hidden layers, taking in the state size as input and outputting the action size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        #Re-initializing the parameter weights by calling the helper function 'hidden_init'
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialization helper function
        """
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Does a forward pass through the Actor nework that maps states -> actions
        """
        x = self.bn1(state)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        #Return a value between -1 and 1
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        
        """
        Initialize parameters and build model
        
        state_size: Environment's state size
        action_size: Agents' action size
        seed: Random seed
        """
        self.seed = torch.manual_seed(seed)
        
        #Initialize batch normalization
        self.bn1 = nn.BatchNorm1d(state_size)
        
        #Initializing hidden layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+2*action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialization helper function
        """
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action, action_player2):
        """
        Does a forward pass through the Critic nework that maps (state, action, action_player2) pairs -> Q-values
        
        state: Current Agent's state
        action: Current Agent's action
        action_player2: Current second Agent's action
        """
        xs = self.bn1(state)
        xs = F.leaky_relu(self.fc1(xs))
        x = torch.cat((xs, action, action_player2), dim=1)
        x = F.leaky_relu(self.fc2(x))
        
        return self.fc3(x)