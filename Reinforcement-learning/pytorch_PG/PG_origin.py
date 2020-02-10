import torch
from torch.distributions import Categorical
import numpy as np
import gym

learning_rate = 0.01
gamma = 0.99
env = gym.make('CartPole-v1')
env.seed(1); 
torch.manual_seed(1);

class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy,self).__init__();
        self.state_space = 4;
        self.action_space = 2;

        self.l1 = torch.nn.Linear(self.state_space,128,bias=False);
        self.l2 = torch.nn.Linear(128,self.action_space,bias=False);
        
        self.dropout = torch.nn.Dropout(p=0.6);
        
        self.ac1 = torch.nn.ReLU();
        self.ac2 = torch.nn.Softmax();

        self.gamma = gamma

        self.policy_history = torch.autograd.Variable(torch.Tensor());

        self.reward_episode = [];
        self.reward_history = [];
        self.loss_history = [];
    
    def forward(self,x):
        x=self.l1(x);
        x=self.dropout(x);
        x=self.ac1(x);

        x=self.ac2(self.l2(x));
        return x;


policy = Policy();
optimizer = torch.optim.Adam(policy.parameters(),lr=learning_rate);


def select_action(state):

    state = torch.from_numpy(state).type(torch.FloatTensor);
    state = policy(torch.autograd.Variable(state));
    c = Categorical(state);# stack state data => a = [[1,2],[3,4],[6,5]] ,c = Categorical(a) 
    action = c.sample();# .sample() return max probability index => c.sample() = [1,1,0]
    #print(policy.policy_history.dim())

    if len(policy.policy_history) > 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).reshape(1)])
        # stack prob by model
    else:
        policy.policy_history = (c.log_prob(action)).reshape(1);
        # .log_prob(index_list) return ln(probability value) accoding to "c.sample()=[1,1,0]" => c.log_prob(c.sample()) = [2,4,6]
    
    return action;


def update_policy():
    R = 0;
    rewards = []

    #print(len(policy.reward_episode))

    # reward stack => reward_episode[1.0,1.0,1.0] => rewards[1.0, 1.99, 2.98]
    for r in policy.reward_episode[::-1]:
        R = r + (policy.gamma * R);
        rewards.insert(0,R);

    # Scale rewards normalization => rewards range(-x,y) => range(-n,n), mean = 0
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards-rewards.mean())/(rewards.std() + np.finfo(np.float32).eps);
    
    # Calculate loss
    pi_r = torch.mul(policy.policy_history,torch.autograd.Variable(rewards));
    loss = -torch.sum(pi_r,dim=-1);

    optimizer.zero_grad()
    loss.backward();
    optimizer.step();
    
    policy.loss_history.append( loss.item() );
    policy.reward_history.append(np.sum(policy.reward_episode));
    
    #initial buffer
    policy.policy_history = torch.autograd.Variable(torch.Tensor())
    policy.reward_episode= []


def main(episodes):

    running_reward = 10

    for episode in range(episodes):
        state = env.reset();
        done = False;
        
        for time in range(1000):
            
            action = select_action(state);
            state,reward,done,_ = env.step(action.item());
            
            policy.reward_episode.append(reward);
            if done:
                break;
            
            if episode % 50 == 0:
                env.render(mode='rgb_array');

        
        running_reward = (running_reward * 0.99) + (time * 0.01);

        update_policy();

        if episode % 50 == 0:
            print("policy.policy_history length : ",len(policy.policy_history));
            print('Episode {}\t Last length:{:5d}\t Average length: {:.2f}'.format(episode,time,running_reward));
        
        if running_reward > env.spec.reward_threshold:
            print("Solved ! Running reward is now {} and the last episode runs to {} time steps! ".format(running_reward,time));

            break;

episodes = 101
main(episodes)

env.close();
