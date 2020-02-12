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
env._max_episode_steps=10001;
torch.manual_seed(543)

SavedAction = namedtuple('SavedAction',['log_prob','value']);

class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy,self).__init__();
        input_size=4
        output_size=2

        self.affine1 = torch.nn.Linear(input_size,128);

        # actor's layer
        self.action_head = torch.nn.Linear(128,output_size);
        
        # critic's layer
        self.value_head = torch.nn.Linear(128,1);

        self.saved_actions =[];
        self.rewards =[];

        self.ac1 = torch.nn.ReLU();
        self.ac2 = torch.nn.Softmax(dim=-1);
    
    def forward(self,x):

        x1 = self.ac1(self.affine1(x));

        action_prob = self.ac2(self.action_head(x1));

        stata_values = self.value_head(x1);

        return action_prob,stata_values;

model = Policy();
optimizer = torch.optim.Adam(model.parameters(),lr=0.03);
eps = np.finfo(np.float32).eps.item();

def select_action(state):
    state = torch.from_numpy(state).float();
    probs, state_value = model(state)

    m = Categorical(probs);

    action = m.sample();

    model.saved_actions.append(SavedAction(m.log_prob(action),state_value));

    return action.item();

def finish_episode():

    R = 0;
    saved_actions = model.saved_actions;

    policy_losses = [];
    value_losses = [];
    returns =[];

    gammma = 0.99;
    for r in model.rewards[::-1]:
        R = r + gammma*R;
        returns.insert(0,R);
    
    returns = torch.tensor(returns);
    returns = (returns-returns.mean()) / (returns.std()-eps);
    

    for (log_prob,value), R_ in zip(saved_actions, returns):

        advantage = R_ - value.item();

        # calculate actor loss
        policy_losses.append(-log_prob*advantage);

        value_losses.append(F.smooth_l1_loss(value,torch.tensor([R_])));
    
    
    optimizer.zero_grad();

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum();

    loss.backward();
    optimizer.step();

    del model.rewards[:]
    del model.saved_actions[:]

def main():
    running_reward = 10;

    for i_episode in count(1):

        state = env.reset();
        ep_reward =0;

        for t in range(1,10000):
            action = select_action(state);

            state, reward,done,_ =env.step(action);

            #if i_episode%50==0:
            env.render();
            
            model.rewards.append(reward);
            ep_reward += reward;
            if done:
                break;
        
        running_reward = 0.05*ep_reward + (1-0.05)*running_reward

        finish_episode();

        # log results
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main();
    env.close();
    


# REFRENCE: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py