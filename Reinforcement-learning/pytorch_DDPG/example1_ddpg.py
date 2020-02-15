import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DKRL
from torch.autograd import Variable
import gc
import matplotlib.pyplot as plt
EPS = 0.003
BATCH_SIZE = 128
GAMMA = 0.99

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__();

        self.state_dim = state_dim;
        self.action_dim = action_dim;

        self.state_fc = nn.Linear(state_dim,256);
        self.state_fc.weight.data = fanin_init(self.state_fc.weight.data.size());

        self.state_fc2 = nn.Linear(256,128)
        self.state_fc2.weight.data = fanin_init(self.state_fc2.weight.data.size());
        
        self.action_fc = nn.Linear(action_dim,128);
        self.action_fc.weight.data = fanin_init(self.action_fc.weight.data.size());


        self.concat_fc = nn.Linear(256,128);
        self.concat_fc.weight.data = fanin_init(self.concat_fc.weight.data.size());

        self.value_fc = nn.Linear(128,1);
        self.value_fc.weight.data.uniform_(-EPS,EPS)

    
    def forward(self,state,action):
        s1 = F.relu(self.state_fc(state));
        s2 = F.relu(self.state_fc2(s1));

        a1 = F.relu(self.action_fc(action));

        
        x = torch.cat((s2,a1),dim=-1);

        x = F.relu(self.concat_fc(x));
        x = self.value_fc(x);
        return x;
    
class Actor(nn.Module):

    def __init__(self,state_dim,action_dim,action_lim):
        super(Actor,self).__init__();

        self.state_dim = state_dim;
        self.action_dim = action_dim;
        self.action_lim = action_lim;

        self.fc1 = nn.Linear(state_dim,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size());
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size());
        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size());
        self.fc4 = nn.Linear(64,action_dim);
        self.fc4.weight.data.uniform_(-EPS,EPS);
        
    def forward(self,state):

        x = F.relu(self.fc1(state));
        x = F.relu(self.fc2(x));
        x = F.relu(self.fc3(x));
        action = F.tanh(self.fc4(x));

        action = action *self.action_lim
        return action;

class OrnsteinUhlenbeckActionNoise:
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

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.clone().data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
			target_param.clone().data * (1.0 - tau) + param.clone().data * tau
		)

class Trainer:
    def __init__(self,state_dim,action_dim,actiom_lim,ram):
        LEARNING_RATE = 0.001;
        self.stata_dim = state_dim;
        self.action_dim = action_dim;
        self.actiom_lim = actiom_lim;
        self.ram = ram;
        self.iter = 0;
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim);
        
        self.actor = Actor(self.stata_dim,self.action_dim,self.actiom_lim).cuda();
        self.target_actor = Actor(self.stata_dim,self.action_dim,self.actiom_lim).cuda();
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE);
        self.target_actor.eval();
        
        self.critic = Critic(self.stata_dim,self.action_dim).cuda();
        self.target_critic = Critic(self.stata_dim,self.action_dim).cuda();
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE);
        self.target_critic.eval();

        hard_update(self.target_actor,self.actor);
        hard_update(self.target_critic,self.critic);
    
    def get_exploitation_action(self,state):
        state = Variable(torch.from_numpy(state)).cuda();
        action = self.target_actor.forward(state).detach();
        return action.item();

    def get_exploration_action(self,state):# add noise
        state = Variable(torch.from_numpy(state)).cuda();
        action = self.actor.forward(state).detach();
        new_action = action.item() + (self.noise.sample()*self.actiom_lim)
        return new_action;
    
    def optimize(self):
        self.actor.train();
        self.critic.train();

        transition = self.ram.sample(BATCH_SIZE)
        batch_data = DKRL.ContinuousSpace.Transition(*zip(*transition));

        # s1=batch_data.state
        # a1=batch_data.action
        # r1=batch_data.reward
        # s2=batch_data.next_state

        s1 = np.vstack(batch_data.state);
        a1 = np.vstack(batch_data.action);
        r1 = np.vstack(batch_data.reward);
        s2 = np.vstack(batch_data.next_state);
        # print(s1.shape);
        # print(a1.shape);
        # print(r1.shape);
        # print(s2.shape);

        s1 = Variable(torch.from_numpy(s1).type(torch.float32)).cuda();
        a1 = Variable(torch.from_numpy(a1).type(torch.float32)).cuda();
        r1 = Variable(torch.from_numpy(r1).type(torch.float32)).cuda();
        s2 = Variable(torch.from_numpy(s2).type(torch.float32)).cuda();

        #============ optimize critic ==================
        self.critic.zero_grad();
        a2 =self.target_actor.forward(s2).detach();
        
        next_val = torch.squeeze(self.target_critic.forward(s2,a2).detach());
            
        y_expected  = torch.squeeze(r1) + GAMMA*next_val;# y_exp = r + gamma*Q'( s2, pi'(s2))
        #y_expected=y_expected.unsqueeze(-1);

        y_predicted = torch.squeeze(self.critic.forward(s1,a1));
        # critic loss = r+gamma*Q(s',a') -Q(s,a);
        loss_critic = F.mse_loss(y_predicted,y_expected);
        
        self.critic_optimizer.zero_grad();
        loss_critic.backward();
        self.critic_optimizer.step();
        
        #============ optimize critic ==================

        #============ optimize actor ===================
        self.actor.zero_grad();
        pred_ac1 = self.actor.forward(s1);
        
        # actor loss = -Q(s,A(s));
        loss_actor = -self.critic.forward(s1,pred_ac1);
        self.actor_optimizer.zero_grad();
        loss_actor = loss_actor.mean();
        loss_actor.backward();
        self.actor_optimizer.step();
        #============ optimize actor ===================

        self.actor.eval();
        self.critic.eval();

        TAU = 0.001;
        soft_update(self.target_actor,self.actor,TAU);
        soft_update(self.target_critic,self.critic,TAU);
        




env = gym.make('Pendulum-v0')

MAX_EPS = 5000;
MAX_STEP = 500;
MAX_BUF = 6000000;
MAX_TOTAL_REWARD = 300;

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

memory=DKRL.ContinuousSpace.ReplayMemory(MAX_BUF);

trainer=Trainer(S_DIM,A_DIM,A_MAX,memory);


scores = [];

for _ep in range(MAX_EPS):
    observation = env.reset();


    print("EPISODE: ",_ep);
    print(observation)
    score= 0.;
    for r in range(MAX_STEP):
        if _ep > 100:
            env.render();

        state = np.float32(observation);

        action = trainer.get_exploration_action(state);

        new_observation,reward, done, info = env.step(action);

        score+=reward;
        if done: 
            new_state = None
        else:
            new_state = np.float32(new_observation);
            memory.push(state,action,new_state,reward);

        
        
        observation = new_observation;
        
        if len(memory)>BATCH_SIZE:
            trainer.optimize();
        
        if done:
            break;
    scores.append(score)
    print("score: ",score);
    print("len(memory): ",len(memory));
    print("=====================")
    
    gc.collect();

env.close();

plt.title('Training....');
plt.xlabel('Episode');
plt.ylabel('Score');
plt.plot(np.array(scores));
plt.show()


# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL