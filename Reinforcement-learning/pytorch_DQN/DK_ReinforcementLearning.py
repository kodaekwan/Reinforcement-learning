
import torch
import random
import numpy as np
import math
from collections import namedtuple
from torch.distributions import Categorical
import cv2

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


Transition = namedtuple('Transition',('state','action','next_state','reward'));

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

class PG_Module():
    #Policy_Gradient
    def __init__(self,policy_net,device=None):
        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.policy_net = policy_net;
        self.policy_net.to(self.device)
        self.policy_net.eval();
        
        # initial policy and reward history
        self.policy_history = torch.autograd.Variable(torch.Tensor());
        self.reward_history = [];

        
    def get_policy_action(self,state,action_num=2):
        # state is network 1 batch input data
        
        output = self.policy_net(torch.autograd.Variable(state).to(self.device));
        c = Categorical(output);# stack output data 
        # example) output = [[1,2],[3,4],[6,5]] ,c = Categorical(output), c = [[1,2],[3,4],[6,5]]

        action = c.sample();# .sample() return max probability index 
        # example) c = [[1,2,0],[3,4,0],[6,5,0],[7,8,9]], c.sample() = [1,1,0,2]

        # stack prob by model
        if len(self.policy_history) > 0:
            self.policy_history = torch.cat([self.policy_history, c.log_prob(action).reshape(1)])
            # .log_prob(index_list) return ln(probability value) accoding to "c.sample()=[1,1,0]" => 
            # example) c = [[1,2,0],[3,4,0],[6,5,0],[7,8,9]], c.sample() = [1,1,0,2], c.log_prob(c.sample()) = [2,4,6,9]
        else:
            self.policy_history = (c.log_prob(action)).reshape(1);
        
        return action;

    def set_Optimizer(self,optimizer=None):
        if(optimizer==None):
            self.optimizer = torch.optim.Adam(self.policy_net.parameters());
        else:
            self.optimizer = optimizer;

    def stack_reward(self,reward=None):
        self.reward_history.append(reward);
    
    def update(self,GAMMA=0.99,parameter_clamp=None):
        # "parameter_clamp" example) parameter_clamp=(-1,1)
        
        if(len(self.reward_history)==0):
            return;

        self.policy_net.train();
        R = 0;
        rewards_trajectory = []

        # accumulate reward  => reward_history[1.0,1.0,1.0] => rewards_trajectory[1.0, 1.99, 2.98]
        for r in self.reward_history[::-1]:
            R = r + (GAMMA * R);
            rewards_trajectory.insert(0,R);

        # Scale rewards_trajectory normalization => rewards_trajectory range(-x,y) => range(-n,n), mean = 0
        rewards_trajectory = torch.FloatTensor(rewards_trajectory);
        rewards_trajectory = (rewards_trajectory-rewards_trajectory.mean())/(rewards_trajectory.std() + np.finfo(np.float32).eps);
        
        # Calculate loss
        # "policy_history" is action trajectory of model.
        # "rewards_trajectory" is result according to action of model.
        pi_r = torch.mul(self.policy_history.to(self.device),torch.autograd.Variable(rewards_trajectory).to(self.device));
        loss = -torch.sum(pi_r,dim=-1);
        
        # update model weight
        self.optimizer.zero_grad();
        loss.backward();
        if(parameter_clamp!=None):
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(parameter_clamp[0],parameter_clamp[1]);
        self.optimizer.step();
        self.policy_net.eval();

        total_reward = np.sum(self.reward_history);
        avg_loss = loss.item();

        # clear policy and reward history
        self.policy_history = torch.autograd.Variable(torch.Tensor());
        self.reward_history= [];
        
        return avg_loss,total_reward;


class DQL_Module():
    # Deep Q learning
    def __init__(self,policy_net,target_net,device=None,batch_size=128,train_start=0):
        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.batch_size = batch_size;
        self.train_start =train_start;
        self.policy_net=policy_net.to(self.device);
        self.target_net=target_net.to(self.device);
        self.target_update();#first, copied weight of policy_net to target net.
        self.target_net.eval();# 
        self.policy_net.eval();# 
        self.target_updata_count = 0;

    def set_Criterion(self,criterion=None):    
        if(criterion==None):
            self.criterion = torch.nn.SmoothL1Loss().to(self.device);
        else:
            self.criterion = criterion.to(self.device);
    
    def set_Optimizer(self,optimizer=None):
        
        if(optimizer==None):
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters());
        else:
            self.optimizer = optimizer;

    def set_Threshold(self,EPS_START=0.9,EPS_END=0.05,EPS_DECAY=200):
        self.Threshold = Episode_Threshold(EPS_START=EPS_START,EPS_END=EPS_END,EPS_DECAY=EPS_DECAY);
    
    def set_Memory(self,capacity=5000,buffer_device=None):
        self.memory = ReplayMemory(capacity);
        
        if(buffer_device==None):
            self.buffer_device=self.device;
        else:
            self.buffer_device=buffer_device;
    
    def get_policy_action(self,state,action_num=2):
        # state is network 1 batch input data
        sample = random.random();
        threshold=self.Threshold.get_threshold();
        if (sample > threshold):
            with torch.no_grad():# don't update 
                output  = self.policy_net(state.to(self.device));
                # max(1) mean data get maximun value based 1 dimension tensor.
                # .max(1) same method => np.argmax(output,axis=1)
                # if a=[[1,2],[3,4],[5,6]], np.argmax(a,axis=1)  = [ 2 , 4 , 6 ];
                index_output=output.max(1)[1];
                # max_value,max_value_index=output.max(1);
                # output.max(1) return tensor[ max value, index of max value]; => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = tensor[[ 2 , 4 , 6 ], [ 1 , 1 , 0 ]]
                # output.max(1)[1] is index => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = [[ 2 , 4 , 6 ], [ 1 , 1 , 0 ] ], a.max(1)[1] = [1,1,0]
                return index_output.view(1,1);
        else:
            return torch.tensor([[random.randrange(action_num)]],device=self.device,dtype=torch.long);
    
    def stack_memory(self,state=None,action=None,next_state=None,reward=None):
        self.memory.push(   state.to(self.buffer_device),
                            action.to(self.buffer_device),
                            next_state if next_state==None else next_state.to(self.buffer_device),
                            reward.to(self.buffer_device));
                            
    def target_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict());#copied weight of policy_net to target net.

    def update(self,GAMMA=0.999,parameter_clamp=None):
        # "parameter_clamp" example) parameter_clamp=(-1,1)

        # if memory size under BATCH_SIZE then  the results continuous stack to memory.
        if len(self.memory)<(self.batch_size+self.train_start):
            return;
        #print("train");
        self.policy_net.train();# 
        # if memory size over self.batch_size then train network model.
        
        # get batch data
        transition = self.memory.sample(self.batch_size);
        batch_data = Transition(*zip(*transition));
        # last data dropout
        non_final_mask = torch.tensor(  tuple(map(lambda s: s is not None, batch_data.next_state))
                                        ,device=self.device
                                        ,dtype=torch.bool);
        non_final_next_state = torch.cat([s for s in batch_data.next_state if s is not None]).to(self.device);

        state_batch = torch.cat(batch_data.state).to(self.device);
        action_batch = torch.cat(batch_data.action).to(self.device);
        reward_batch = torch.cat(batch_data.reward).to(self.device);

        state_action_values = self.policy_net(state_batch).gather(1,action_batch);
        # gather() parsing the result according to action_batch.
        # example) a=[[1,2],[3,4],[5,6]],b=[1,0,1] => a.gather(1,b) = [2,3,6]
        
        next_state_values = torch.zeros(self.batch_size,device=self.device);
        next_state_values[non_final_mask]  = self.target_net(non_final_next_state).max(1)[0].detach();# target_net decide next state. In order to separated update, you detach the result.
        
        expected_state_action_values = (next_state_values*GAMMA) + reward_batch;

        loss = self.criterion(state_action_values,expected_state_action_values.unsqueeze(1));
        #loss = F.mse_loss(state_action_values,expected_state_action_values.unsqueeze(1));
        
        self.optimizer.zero_grad();
        loss.backward();

        if(parameter_clamp!=None):
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(parameter_clamp[0],parameter_clamp[1]);

        self.optimizer.step();
        self.policy_net.eval();# 




class GAME():
    def __init__(self,game_name='CartPole-v0'):
        import gym #$ pip3 install gym
        self.env = gym.make(game_name);# make game
        self.reset();
        
    def get_cart_location(self,image_width):
        # you can get cart position from pixel unit
        # env.state[0] is cart position
        world_width = self.env.x_threshold * 2;
        scale = image_width/world_width;
        return int(self.env.state[0]*scale + image_width/2.0);
    
    def reset(self):
        self.env.reset();
        image_shape = self.get_screen().shape;
        self.max_key_num = self.env.action_space.n;
        self.image_height = image_shape[0];
        self.image_width = image_shape[1];
        self.image_channel = image_shape[2];
        self.cart_location=self.get_cart_location(self.image_width);
    
    def get_screen(self):
        return self.env.render(mode='rgb_array');
    
    def set_control(self,key):
        return self.env.step(key);

    def close(self):
        self.env.close();
    
    def cut_image(self,src,x=0,y=0,width=600,height=400):
        #src (H,W,C)
        return src[y:y+height,x:x+width:];
    
    def focus_cut_image(self,src,focus=(0,1),width=600,height=400):
        # "src" is numpy type and have shape to (H,W,C)
        # "focus" is (x,y)
        # "width" is cutting range from "focus x"
        # "height" is cutting range from "focus y".
        screen_shape=src.shape;
        
        screen_height=screen_shape[0];
        screen_width=screen_shape[1];

        if focus[0] < width//2:
            w_slice_range = slice(0,width)
        elif focus[0] > (screen_width - (width//2) ):
            w_slice_range = slice(-width,None);
        else:
            w_slice_range = slice((focus[0]-(width//2)), (focus[0]+(width//2)));
        
        if focus[1] < height//2:
            h_slice_range = slice(0,height)
        elif focus[1] > (screen_height - (height//2) ):
            h_slice_range = slice(-height,None);
        else:
            h_slice_range = slice((focus[1]-(height//2)), (focus[1]+(height//2)));

        return src[h_slice_range,w_slice_range,:];
    
    def resize_image(self,src,width,height):
        return cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_CUBIC);
