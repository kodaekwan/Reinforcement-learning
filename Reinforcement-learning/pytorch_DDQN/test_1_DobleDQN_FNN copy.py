import gym
import time

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DK_ReinforcementLearning




class Model(torch.nn.Module):
    def __init__(self,input_size=4,output_size=2):
        super(Model,self).__init__();
        
        self.linear1 = torch.nn.Linear(input_size,24);
        torch.nn.init.uniform_(self.linear1.weight);
        self.norm1 = torch.nn.BatchNorm1d(24);
        self.ac1  = torch.nn.ReLU();
    
        self.linear2 = torch.nn.Linear(24,24);
        torch.nn.init.uniform_(self.linear2.weight);
        self.norm2 = torch.nn.BatchNorm1d(24);
        self.ac2  = torch.nn.ReLU();
    
        self.head = torch.nn.Linear(24,output_size);
        torch.nn.init.uniform_(self.head.weight);

    def forward(self,x):
        x=self.linear1(x);
        x=self.norm1(x);
        x=self.ac1(x);

        x=self.linear2(x);        
        x=self.norm2(x);
        x=self.ac2(x);

        return self.head(x.view(x.size(0),-1));#(batch,2)


game=DK_ReinforcementLearning.GAME();
game.env._max_episode_steps=10001;

max_action_num=game.max_key_num;
print("game action number : ",max_action_num);


device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
policy_net=Model(input_size=4,output_size=2);
target_net=Model(input_size=4,output_size=2);

RL=DK_ReinforcementLearning.DDQN_Module(policy_net=policy_net,
                            target_net=target_net,
                            device=device,
                            batch_size=64,
                            train_start=936);

RL.set_Criterion(criterion=torch.nn.MSELoss())
RL.set_Optimizer(optimizer=torch.optim.Adam(RL.policy_net.parameters(),lr=0.001,weight_decay=0.01));
RL.set_Threshold(EPS_START=1.0,EPS_END=0.01,EPS_DECAY=1000);
RL.set_Memory(capacity=2000,buffer_device=torch.device("cpu"));


for episode in range(1000):
    #view
    game.reset();
    observation = game.env.state;
    now_state = torch.tensor([observation],dtype=torch.float32);
    score = 0

    for t  in range(1000):

        # Decide action from policy network
        action=RL.get_policy_action(state=now_state,action_num=2);

        # Execute action in Game environment by network policy
        observation,reward,done,info = game.set_control(action.item());
        # observation shape => [cart-position ,cart-velocity, pole-position, pole-velocity]

        reward = reward if not done or score >= 499 else -100.0
        score += reward

        # !!view current image but not using to train!!
        if episode%50==0:
            now_screen = game.get_screen();

        if not done:
            next_state = torch.tensor([observation],dtype=torch.float32);
        else:
            next_state = None;

        # stack results to memory
        RL.stack_memory(now_state,action,next_state,torch.tensor([reward]));
        
        # change from now data to previous data by time flow.
        now_state = next_state;
        
        if(done==True):
            #print(observation)
            print("episode: ",episode,", done :",t)
            if(t>100):
                print("==================100=======")
            break;
        
        # update policy model
        RL.update(GAMMA=0.99,parameter_clamp=(-1,1));

    # target model synchronization with policy model.
    if episode%10==0:
        RL.target_update();


game.close();
plt.close();