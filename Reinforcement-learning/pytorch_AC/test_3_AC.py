import time
import numpy as np
import torch
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DKRL

torch.manual_seed(1);

class Actor(torch.nn.Module):
    def __init__(self,input_size=4,output_size=2):
        super(Actor,self).__init__();
        self.input_size=input_size
        self.output_size=output_size

        self.linear1 = torch.nn.Linear(self.input_size,64,bias=False);
        self.ac1  = torch.nn.ReLU();
        self.dropout1 = torch.nn.Dropout(0.1);
    
        self.linear2 = torch.nn.Linear(64,64,bias=False);
        self.ac2  = torch.nn.ReLU();
        self.dropout2 = torch.nn.Dropout(0.1);
    
        self.classifier = torch.nn.Linear(64,self.output_size);
        self.classifier.weight.data.uniform_(-3e-3, 3e-3);

    def forward(self,state):
        x=state
        x=self.linear1(x);
        x=self.dropout1(x);
        x=self.ac1(x);

        x=self.linear2(x);        
        x=self.dropout2(x);
        x=self.ac2(x);
        return self.classifier(x);


class Critic(torch.nn.Module):
    def __init__(self,input_size=4):
        super(Critic,self).__init__();
        
        self.linear1 = torch.nn.Linear(input_size,64,bias=False);
        self.ac1  = torch.nn.ReLU();
        self.dropout1 = torch.nn.Dropout(0.1);
    
        self.linear2 = torch.nn.Linear(64,64,bias=False);
        self.ac2  = torch.nn.ReLU();
        self.dropout2 = torch.nn.Dropout(0.1);
    
        self.classifier = torch.nn.Linear(64,1);
        self.classifier.weight.data.uniform_(-3e-3, 3e-3);

    def forward(self,state,action):

        x=torch.cat([state,action],-1);
        x=self.linear1(x);
        x=self.dropout1(x);
        x=self.ac1(x);

        x=self.linear2(x);        
        x=self.dropout2(x);
        x=self.ac2(x);
        return self.classifier(x);

game=DKRL.GAME('CartPole-v0');
game.env._max_episode_steps=10001;
game.env.seed(1); 

max_action_num=game.max_key_num;
print("game action number : ",max_action_num);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

Actor_net=Actor(input_size=4,output_size=2);
Critic_net=Critic(input_size=Actor_net.input_size+Actor_net.output_size);

RL=DKRL.DiscreteSpace.AC_PG_Module(Actor_net=Actor_net,
                                                Critic_net=Critic_net,
                                                device=device,
                                                using_entropy=False);

RL.set_ActorOptimizer(optimizer=torch.optim.Adam(RL.Actor_net.parameters(),lr=0.01));
RL.set_CriticOptimizer(optimizer=torch.optim.Adam(RL.Critic_net.parameters(),lr=0.01));
RL.set_Criterion(criterion=torch.nn.SmoothL1Loss());

for episode in range(1000):
    #view
    game.reset();
    now_state = game.env.state;
    for t  in range(1000):

        # Decide action from policy network and !!!stack policy ouput to memory!!!
        action=RL.get_policy_action(state=now_state,action_num=None);
        
        # Execute action in Game environment by network policy
        observation,reward,done,info = game.set_control(action);

        # observation shape => [cart-position ,cart-velocity, pole-position, pole-velocity]
        # stack results to memory
        RL.stack_reward(reward);

        now_state=observation;

        if episode%50==0:
            # !!view current image but not using to train!!
            now_screen = game.get_screen();

        if(done==True):
            #print(observation)
            print("episode: ",episode,", done :",t)
            if(t>100):
                print("==================100=======")
            break;
    
    # update 
    RL.update(GAMMA=0.99,parameter_clamp=None);

game.close();
