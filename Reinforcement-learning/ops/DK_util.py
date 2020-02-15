import torch
import random
import numpy as np
import math
from collections import namedtuple
from torch.distributions import Categorical
from torch.distributions import Normal

Transition = namedtuple('Transition',('state','action','next_state','reward'));

History = namedtuple('History',['log_prob','value']);

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity;
        self.memory = [];
        self.position = 0;
    
    def push(self,*args):
        """ Save as transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None);
        self.memory[self.position] = Transition(*args);
        self.position = (self.position + 1) % self.capacity;#if capacity is 1000 and position is 0 then (0+1)%1000= 1, so this algorithm Queue

    def sample(self, batch_size):
        return random.sample(self.memory,batch_size);

    def __len__(self):
        return len(self.memory);

class Episode_Threshold():#random Threshold
    def __init__(self,EPS_START = 0.9,EPS_END = 0.05,EPS_DECAY = 200):
        self.EPS_START=EPS_START;
        self.EPS_END=EPS_END;
        self.EPS_DECAY=EPS_DECAY;
        self.step=0;
    def get_threshold(self):
        eps_threshold = self.EPS_END+ (self.EPS_START-self.EPS_END)*math.exp(-1. * self.step / self.EPS_DECAY);
        self.step+=1;
        return eps_threshold;

class OrnsteinUhlenbeckActionNoise:
	"""
		Ornstein–Uhlenbeck process
		Ornstein–Uhlenbeck process is Wiener process about random walk with linear drift term.
		https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
		
		Ornstein–Uhlenbeck term
			dxt = θ( μ − xt )dt + σ dWt
			where θ==theta,μ == mu,σ==sigma is a constant, θ > 0 and σ > 0 are parameters, Wt denotes the Wiener process.
		
		Linear Drift term
			lineardrift = θ( μ − xt )dt
			where dt is 1 in this class.
	"""
	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

def hard_update(target,source):
	"""
	Copies the parameters from source network to target network
	target: Target network (PyTorch)
	source: Source network (PyTorch)
	return: empty
	===================================================================================
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

	Original hard update method 
	===================================================================================
	I tried to follow update method of the reference code. however I couldn't update weight of model in now pytorch. so I developed to trick using model load method.
	the reference code : https://github.com/vy007vikas/PyTorch-ActorCriticRL.
	"""
	target_temp = target.state_dict();
	source_temp = source.state_dict();
	for key in source_temp.keys():
		if key in target_temp.keys():
			target_temp[key].copy_(source_temp[key]);
	target.load_state_dict(target_temp);
	
def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	target: Target network (PyTorch)
	source: Source network (PyTorch)
	tau   : Soft Update Rate == Low Pass Filter Gain 
	return: empty
	===================================================================================
	for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)
	Original soft update method 
	===================================================================================
	I tried to follow update method of the reference code. however I couldn't update weight of model in now pytorch. so I developed to trick using model load method.
	the reference code : https://github.com/vy007vikas/PyTorch-ActorCriticRL.
	
	"""
	target_temp = target.state_dict();
	source_temp = source.state_dict();
	for key in source_temp.keys():
		if key in target_temp.keys():
			target_temp[key].copy_( ((1.0 - tau)*target_temp[key]) + (tau*source_temp[key]));
	target.load_state_dict(target_temp);
	
