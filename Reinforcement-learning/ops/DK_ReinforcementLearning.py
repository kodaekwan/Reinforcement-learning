
import torch
import random
import numpy as np
import math
from collections import namedtuple
from torch.distributions import Categorical


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

class AC_PG_Module():
    # Actor-Critic Stereo type Policy Gradient
    # Stereo mean that Separated Actor and Critic layer.
    #   Input     [State]       [State]
    #                ▽             ▽
    #          |===========| |===========|
    #          | [Decoder] | | [Decoder] |
    #  Network |     ▽     | |     ▽     |
    #          |  [Actor]  | | [Critic]  |
    #          |===========| |===========|
    #               ▽              ▽    
    # Output    [Advantage],     [Value]    !! Advantage channel size is policy number,  Value channel size is 1 !!
    
    def __init__(self,Actor_net,Critic_net,device=None):
        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.Actor_net = Actor_net;
        self.Actor_net.to(self.device)
        self.Actor_net.eval();

        self.Critic_net = Critic_net;
        self.Critic_net.to(self.device)
        self.Critic_net.eval();

        self.history =[];
        self.rewards =[];
        self.softmax = torch.nn.Softmax().to(self.device);
        self.eps = np.finfo(np.float32).eps.item();

    def set_Criterion(self,criterion=None):    
        if(criterion==None):
            self.criterion = torch.nn.SmoothL1Loss().to(self.device);
        else:
            self.criterion = criterion.to(self.device);

    def set_ActorOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.Actor_optimizer = torch.optim.Adam(self.Actor_net.parameters());
        else:
            self.Actor_optimizer = optimizer;
    
    def set_CriticOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.Critic_optimizer = torch.optim.Adam(self.Critic_net.parameters());
        else:
            self.Critic_optimizer = optimizer;

    def get_policy_action(self,state,action_num=None):
        
        #state = torch.tensor([state],dtype=torch.float32).to(self.device);
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0);
        state_value =self.Critic_net(state);
        probs = self.Actor_net(state);
        probs = self.softmax(probs);
        m = Categorical(probs);
        action = m.sample();
        self.history.append( History( m.log_prob(action),state_value ) );
        return action.item();
    
    def stack_reward(self,reward=None):
        self.rewards.append(reward);
    
    def update(self,GAMMA=0.99,parameter_clamp=None):
        # "parameter_clamp" example) parameter_clamp=(-1,1)
        
        if(len(self.rewards)==0):
            return;
        self.Critic_net.train();
        self.Actor_net.train();
        R = 0;

        policy_losses = [];
        value_losses = [];
        returns =[];

        for r in self.rewards[::-1]:
            R = r + GAMMA*R;
            returns.insert(0,R);
        
        returns = torch.tensor(returns);
        returns = (returns-returns.mean()) / (returns.std()-self.eps);
        
        # Calculate loss
        # Monte-Carlo Policy Gradient
        # Loss = − Qp(s,a)logπ(a|s)
        # Qp        : policy network
        # Qp(s,a)   : Accumulated total reward -> returns
        # logπ(a|s) : log-probability of the action taken ->log_prob
        for (log_prob,value), returns_ in zip(self.history, returns):
            
            value=value.squeeze(0);
            returns_ = torch.tensor(returns_).to(self.device);
            
            # Advantage Actor-Critic
            # A(s,a)  =  Q(s,a)  −  V(s)
            advantage = returns_ - value;
            
            # policy Loss = -A(s,a)logπ(a|s)
            policy_losses.append(-log_prob*advantage.detach());# In order to Stable calculate, it need to advantage.detach()
            
            # value Loss = Distance between Q(s,a) and V(s)
            value_losses.append(self.criterion(value,returns_));
        

        self.Critic_optimizer.zero_grad();
        critic_loss = torch.stack(value_losses).sum();
        critic_loss.backward();
        self.Critic_optimizer.step();

        self.Actor_optimizer.zero_grad();
        actor_loss = torch.stack(policy_losses).sum();
        actor_loss.backward();
        self.Actor_optimizer.step();

        del self.rewards[:];
        del self.history[:];
        
        self.Critic_net.eval();
        self.Actor_net.eval();
        return actor_loss.item(),critic_loss.item();



class AC_Mono_PG_Module():
    # Actor-Critic Mono type Policy Gradient
    # Mono mean that Comprise Actor-Critic layer in One Network, it share feature through Decoder layer.
    #   Input        [State]
    #                   ▽
    #          |==================|
    #          |    [Decoder]     |
    #  Network |      ▽   ▽       |
    #          | [Actor] [Critic] |
    #          |==================|
    #               ▽        ▽    
    # Output  [[Advantage],[Value]]    !! Advantage channel size is policy number,  Value channel size is 1 !!

    def __init__(self,Actor_Critic_net,device=None):
        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.Actor_Critic_net = Actor_Critic_net;
        self.Actor_Critic_net.to(self.device)
        self.Actor_Critic_net.eval();

        self.history =[];
        self.rewards =[];
        self.softmax = torch.nn.Softmax().to(self.device);
        self.eps = np.finfo(np.float32).eps.item();
    
    def set_Criterion(self,criterion=None):    
        if(criterion==None):
            self.criterion = torch.nn.SmoothL1Loss().to(self.device);
        else:
            self.criterion = criterion.to(self.device);

    def set_Optimizer(self,optimizer=None):
        if(optimizer==None):
            self.optimizer = torch.optim.Adam(self.Actor_Critic_net.parameters());
        else:
            self.optimizer = optimizer;

    def get_policy_action(self,state,action_num=None):
        
        #state = torch.tensor([state],dtype=torch.float32).to(self.device);
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0);
        probs,state_value = self.Actor_Critic_net(state);
        probs=self.softmax(probs);     
        m = Categorical(probs);
        action = m.sample();
        self.history.append( History( m.log_prob(action),state_value ) );
        return action.item();
    
    def stack_reward(self,reward=None):
        self.rewards.append(reward);
    

    def update(self,GAMMA=0.99,parameter_clamp=None):
        # "parameter_clamp" example) parameter_clamp=(-1,1)
        
        if(len(self.rewards)==0):
            return;
        self.Actor_Critic_net.train();
        R = 0;

        policy_losses = [];
        value_losses = [];
        returns =[];

        for r in self.rewards[::-1]:
            R = r + GAMMA*R;
            returns.insert(0,R);
        
        returns = torch.tensor(returns);
        returns = (returns-returns.mean()) / (returns.std()-self.eps);
        
        
        # Calculate loss
        # Monte-Carlo Policy Gradient
        # Loss = − Qp(s,a)logπ(a|s)
        # Qp        : policy network
        # Qp(s,a)   : Accumulated total reward -> returns
        # logπ(a|s) : log-probability of the action taken ->log_prob
        for (log_prob,value), returns_ in zip(self.history, returns):
            
            value=value.squeeze(0);
            returns_ = torch.tensor(returns_).to(self.device);
            
            # Advantage Actor-Critic
            # A(s,a)  =  Q(s,a)  −  V(s)
            advantage = returns_ - value;
            
            # policy Loss = -A(s,a)logπ(a|s)
            policy_losses.append(-log_prob*advantage.detach());# In order to Stable calculate, it need to advantage.detach()
            
            # value Loss = Distance between Q(s,a) and V(s)
            value_losses.append(self.criterion(value,returns_));
        
        
        self.optimizer.zero_grad();
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum();
        loss.backward();
        self.optimizer.step();

        del self.rewards[:];
        del self.history[:];
        
        self.Actor_Critic_net.eval();
        return loss.item();        








class PG_Module():
    # Monte-Carlo Policy_Gradient
    def __init__(self,policy_net,device=None):
        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.policy_net = policy_net;
        self.policy_net.to(self.device)
        self.policy_net.eval();
        
        # initial policy and reward history
        self.history = [];
        self.rewards = [];
        torch.nn.gaussian
        self.softmax = torch.nn.Softmax().to(self.device);
        self.eps = np.finfo(np.float32).eps;

        
    def get_policy_action(self,state,action_num=2):
        # state is network 1 batch input data
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0);
        
        output = self.policy_net(state);
        output = self.softmax(output);

        c = Categorical(output);# stack output data 
        # example) output = [[1,2],[3,4],[6,5]] ,c = Categorical(output), c = [[1,2],[3,4],[6,5]]

        action = c.sample();# .sample() return max probability index 
        # example) c = [[1,2,0],[3,4,0],[6,5,0],[7,8,9]], c.sample() = [1,1,0,2]

        # stack prob by model
        self.history.append( History( c.log_prob(action),None) );
        # .log_prob(index_list) return ln(probability value) accoding to index"c.sample()=[1,1,0]"
            # example) c = [[1,2,0],[3,4,0],[6,5,0],[7,8,9]], c.sample() = [1,1,0,2], c.log_prob(c.sample()) = [ln(2),ln(4),ln(6),ln(9)]
        return action.item();

    def set_Optimizer(self,optimizer=None):
        if(optimizer==None):
            self.optimizer = torch.optim.Adam(self.policy_net.parameters());
        else:
            self.optimizer = optimizer;

    def stack_reward(self,reward=None):
        self.rewards.append(reward);
    
    def update(self,GAMMA=0.99,parameter_clamp=None):
        # "parameter_clamp" example) parameter_clamp=(-1,1)
        
        if(len(self.rewards)==0):
            return;

        self.policy_net.train();

        R = 0;
        returns = []

        # accumulated reward : "returns"=[1.0,1.0,1.0] => "returns"=[2.98, 1.99, 1.0]
        for r in self.rewards[::-1]:
            R = r + (GAMMA * R);
            returns.insert(0,R);

        # Scale accumulated reward normalization => "returns" change the range from (1.0, N) to (-n , n) and mean = 0
        returns = torch.FloatTensor(returns);
        returns = (returns-returns.mean())/(returns.std() + self.eps);
        
        # Calculate loss
        # Monte-Carlo Policy Gradient
        # Loss = − Qp(s,a)logπ(a|s)
        # Qp        : policy network
        # Qp(s,a)   : Accumulated total reward ->returns
        # logπ(a|s) : log-probability of the action taken ->log_prob
        policy_losses = [];
        
        # policy Gradient objective is pi_r maximize.
        for (log_prob,_),returns_ in zip(self.history, returns):
            pi_r = log_prob*returns_;
            policy_losses.append(-pi_r);

        loss = torch.stack(policy_losses).sum();
        # this score function had high variance problem because logπ(a|s).

        # update model weight
        self.optimizer.zero_grad();
        loss.backward();
        if(parameter_clamp!=None):
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(parameter_clamp[0],parameter_clamp[1]);
        self.optimizer.step();
        self.policy_net.eval();

        # clear policy history and reward 
        del self.history[:];
        del self.rewards[:];
        
        return loss.item();


class DQN_Module():
    # Deep Q Networks
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
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0);

        sample = random.random();
        threshold=self.Threshold.get_threshold();

        if (sample > threshold):
            with torch.no_grad():# don't update 
                output  = self.policy_net(state);
                # max(1) mean data get maximun value based 1 dimension tensor.
                # .max(1) same method => np.argmax(output,axis=1)
                # if a=[[1,2],[3,4],[5,6]], np.argmax(a,axis=1)  = [ 2 , 4 , 6 ];
                index_output=output.max(1)[1];
                # max_value,max_value_index=output.max(1);
                # output.max(1) return tensor[ max value, index of max value]; => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = tensor[[ 2 , 4 , 6 ], [ 1 , 1 , 0 ]]
                # output.max(1)[1] is index => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = [[ 2 , 4 , 6 ], [ 1 , 1 , 0 ] ], a.max(1)[1] = [1,1,0]
                return index_output.item();
        else:
            return random.randrange(action_num);
    
    def stack_memory(self,state=None,action=None,next_state=None,reward=None):
        # "state" type numpy array
        # "action" type int
        # "next_state" type numpy array
        # "reward" type flot or int
        self.memory.push(   torch.from_numpy(state).float().unsqueeze(0).to(self.buffer_device),
                            torch.from_numpy(np.array([[action]])).to(self.buffer_device),
                            next_state if next_state is None else torch.from_numpy(next_state).float().unsqueeze(0).to(self.buffer_device),
                            torch.from_numpy(np.array([[reward]])).float().to(self.buffer_device));
    
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
        
        expected_state_action_values = reward_batch.squeeze()+(GAMMA*next_state_values);

        
        loss = self.criterion(state_action_values.squeeze(),expected_state_action_values);
        # loss = reward + gamma*Qt(s',a') - Qp(s,a);
        
        self.optimizer.zero_grad();
        loss.backward();

        if(parameter_clamp!=None):
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(parameter_clamp[0],parameter_clamp[1]);

        self.optimizer.step();
        self.policy_net.eval();# 


class DDQN_Module():
    # Double Deep Q Networks
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
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0);

        sample = random.random();
        threshold=self.Threshold.get_threshold();

        if (sample > threshold):
            with torch.no_grad():# don't update 
                output  = self.policy_net(state);
                # max(1) mean data get maximun value based 1 dimension tensor.
                # .max(1) same method => np.argmax(output,axis=1)
                # if a=[[1,2],[3,4],[5,6]], np.argmax(a,axis=1)  = [ 2 , 4 , 6 ];
                index_output=output.max(1)[1];
                # max_value,max_value_index=output.max(1);
                # output.max(1) return tensor[ max value, index of max value]; => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = tensor[[ 2 , 4 , 6 ], [ 1 , 1 , 0 ]]
                # output.max(1)[1] is index => example a=tensor[[1,2],[3,4],[6,1]], a.max(1) = [[ 2 , 4 , 6 ], [ 1 , 1 , 0 ] ], a.max(1)[1] = [1,1,0]
                return index_output.item();
        else:
            return random.randrange(action_num);
    
    def stack_memory(self,state=None,action=None,next_state=None,reward=None):
        # "state" type numpy array
        # "action" type int
        # "next_state" type numpy array
        # "reward" type flot or int

        self.memory.push(   torch.from_numpy(state).float().unsqueeze(0).to(self.buffer_device),
                            torch.from_numpy(np.array([[action]])).to(self.buffer_device),
                            next_state if next_state is None else torch.from_numpy(next_state).float().unsqueeze(0).to(self.buffer_device),
                            torch.from_numpy(np.array([[reward]])).float().to(self.buffer_device));
                            
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

        # ========================== Double DQN ======================================
        # reference http://papers.nips.cc/paper/3964-double-q-learning
        
        # ****  a' of origin DQN is result of past Qp ****
        # Origin DQN:  loss = reward + gamma*Qt(s',a') - Qp(s,a);
        
        # **** a' of Double DQN is result of current Qp ****
        # Double DQN:  loss = reward + gamma*Qt(s', Qp(s',a') ) - Qp(s,a); 
        
        next_state_action_values = self.policy_net(non_final_next_state).max(1)[1].detach();
        #                   ^ a' = Qp(s',a')
        
        next_state_values = torch.zeros(self.batch_size,device=self.device);
        next_state_values[non_final_mask]  = self.target_net(non_final_next_state).gather(1,next_state_action_values.unsqueeze(1)).detach().squeeze(1);
        #                                              ^ Qt(s', Qp(s',a') )
        
        expected_state_action_values = reward_batch.squeeze()+(GAMMA*next_state_values);
        # ========================== Double DQN ======================================
        
        loss = self.criterion(state_action_values.squeeze(),expected_state_action_values);# "state_action_values" shape is [batch,1] so .squeeze()
        
        self.optimizer.zero_grad();
        loss.backward();

        if(parameter_clamp!=None):
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(parameter_clamp[0],parameter_clamp[1]);

        self.optimizer.step();
        self.policy_net.eval();# 


class GAME():
    import cv2
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
        return self.cv2.resize(src, dsize=(width, height), interpolation=self.cv2.INTER_CUBIC);
