import torch
import numpy as np
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DKRL


class Actor_Model(torch.nn.Module):

    def __init__(self,state_dim,action_dim,output_std=1.0,EPS = 0.003):
        super(Actor_Model,self).__init__();
        self.state_dim = state_dim;
        self.action_dim = action_dim;
        self.output_std = output_std;

        self.fc1 = torch.nn.Linear(state_dim,128);
        torch.nn.init.xavier_normal_(self.fc1.weight.data);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(128,128);
        torch.nn.init.xavier_normal_(self.fc2.weight.data);
        self.ac2 = torch.nn.ReLU();

        self.fc3 = torch.nn.Linear(128,action_dim);
        self.fc3.weight.data.uniform_(-EPS,EPS);
        self.tanh = torch.nn.Tanh();
    
    def forward(self,state):
        x=self.fc1(state);
        x=self.ac1(x);

        x=self.fc2(x);
        x=self.ac2(x);

        x=self.fc3(x);
        output=self.output_std*self.tanh(x);

        return output;


class Critic_Model(torch.nn.Module):

    def __init__(self,state_dim,action_dim,EPS = 0.003):
        super(Critic_Model,self).__init__();
        self.state_dim = state_dim;
        self.action_dim = action_dim;

        self.fc1 = torch.nn.Linear(state_dim,128);
        torch.nn.init.xavier_normal_(self.fc1.weight.data);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(action_dim,128);
        torch.nn.init.xavier_normal_(self.fc2.weight.data);
        self.ac2 = torch.nn.ReLU();

        self.fc3 = torch.nn.Linear(256,128);
        torch.nn.init.xavier_normal_(self.fc3.weight.data);
        self.ac3 = torch.nn.ReLU();

        self.fc4 = torch.nn.Linear(128,1);
        self.fc4.weight.data.uniform_(-EPS,EPS);
    
    def forward(self,state,action):
        s = self.fc1(state);
        s = self.ac1(s);

        a = self.fc2(action);
        a = self.ac2(a);

        sa = torch.cat((s,a),-1);

        sa = self.fc3(sa);
        sa = self.ac3(sa);

        output = self.fc4(sa);
        return output;

game=DKRL.GAME();
game.env._max_episode_steps=10001;
game.env.seed(1); 

max_action_num=game.max_key_num;
print("game action number : ",max_action_num);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

state_dim = 3
action_dim = 2 
action_std = 2.0

Actor_net=Actor_Model(state_dim=state_dim,action_dim=action_dim,output_std=action_std);
target_Actor_net=Actor_Model(state_dim=state_dim,action_dim=action_dim,output_std=action_std);

Critic_net=Critic_Model(state_dim=state_dim,action_dim=action_dim);
target_Critic_net=Critic_Model(state_dim=state_dim,action_dim=action_dim);









