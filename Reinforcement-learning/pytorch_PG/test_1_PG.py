import time
import numpy as np
import torch
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DKRL

torch.manual_seed(1);

class Model(torch.nn.Module):
    def __init__(self,input_size=4,output_size=2):
        super(Model,self).__init__();
        
        self.linear1 = torch.nn.Linear(input_size,64,bias=False);
        self.ac1  = torch.nn.ReLU();
        self.dropout1 = torch.nn.Dropout(0.1);
    
        self.linear2 = torch.nn.Linear(64,64,bias=False);
        self.ac2  = torch.nn.ReLU();
        self.dropout2 = torch.nn.Dropout(0.1);
    
        self.head = torch.nn.Linear(64,output_size,bias=False);

    def forward(self,x):
        x=self.linear1(x);
        x=self.dropout1(x);
        x=self.ac1(x);
        
        x=self.linear2(x);        
        x=self.dropout2(x);
        x=self.ac2(x);
        
        return self.head(x);


game=DKRL.GAME();
game.env._max_episode_steps=10001;
game.env.seed(1); 

max_action_num=game.max_key_num;
print("game action number : ",max_action_num);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

#policy gradient requirement single model
policy_net=Model(input_size=4,output_size=2);

RL=DKRL.DiscreteSpace.PG_Module(  policy_net=policy_net,
                                        device=device);

RL.set_Optimizer(optimizer=torch.optim.Adam(RL.policy_net.parameters(),lr=0.01));

for episode in range(1000):
    #view
    game.reset();
    now_state = game.env.state
    for t  in range(1000):

        # Decide action from policy network and !!!stack policy ouput to memory!!!
        action=RL.get_policy_action(state=now_state,action_num=2);

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
    
    # update policy model
    RL.update(GAMMA=0.99,parameter_clamp=(-1,1));


game.close();
