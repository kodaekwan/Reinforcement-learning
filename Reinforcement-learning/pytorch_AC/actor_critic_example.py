# REFRENCE: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
 

env = gym.make('CartPole-v0')
env.seed(543)
torch.manual_seed(543)

SavedAction = namedtuple('SavedAction',['log_prob','value']);

class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy,self).__init__();

        self.affine1 = torch.nn.Linear(4,128);

        # actor's layer
        self.action_head = torch.nn.Linear(128,2);
