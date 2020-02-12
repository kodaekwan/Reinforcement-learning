import time
import numpy as np
import torch
import sys
sys.path.append("..")
import ops.DK_ReinforcementLearning as DK_ReinforcementLearning
import torch.nn.functional as F



class Model(torch.nn.Module):
    def __init__(self,height,width,output_size):
        super(Model,self).__init__();

        self.conv1 = torch.nn.Conv2d(3,16,kernel_size=5,stride=2);
        self.bn1 = torch.nn.BatchNorm2d(16);
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5,stride=2);
        self.bn2 = torch.nn.BatchNorm2d(32);
        self.conv3 = torch.nn.Conv2d(32,32,kernel_size=5,stride=2);
        self.bn3 = torch.nn.BatchNorm2d(32);
        
        def conv2d_size_out(size, kernel_size=5, stride = 2):
            return ((size - (kernel_size-1)-1) // stride) + 1;

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)));
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)));
        linear_input_size = convw * convh * 32;

        self.head = torch.nn.Linear(linear_input_size,output_size);

    def forward(self,x):
        # if height = 40, width= 90, output_size = 2
        x = F.relu(self.bn1(self.conv1(x)));#(batch,16,40,90)->(batch,16,18,43)
        x = F.relu(self.bn2(self.conv2(x)));#(batch,16,18,43)->(batch,32,7,20)
        x = F.relu(self.bn3(self.conv3(x)));#(batch,32,7,20)->(batch,32,2,8)
        return self.head(x.view(x.size(0),-1));#(batch,32,2,8)-> (batch,32*2*8) -> (batch,2)



#======================== game setting ======================================
game=DK_ReinforcementLearning.GAME(game_name='CartPole-v0');
game.env._max_episode_steps=10001;

image_height=game.image_height;
image_width=game.image_width;
image_channel=game.image_channel;
max_action_num=game.max_key_num;

print("image height : ",image_height);
print("image width : ",image_width);
print("image channel : ",image_channel);
print("game action number : ",max_action_num);
#======================== game setting ======================================

#======================== model create ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
policy_net=Model(height=40,width=90,output_size=2);
target_net=Model(height=40,width=90,output_size=2);

RL=DK_ReinforcementLearning.DQN_Module(policy_net=policy_net,
                            target_net=target_net,
                            device=device,
                            batch_size=128);

RL.set_Criterion(criterion=torch.nn.MSELoss())
RL.set_Optimizer(optimizer=torch.optim.Adam(RL.policy_net.parameters()));
RL.set_Threshold(EPS_START=0.9,EPS_END=0.01,EPS_DECAY=400);
RL.set_Memory(capacity=5000,buffer_device=torch.device("cpu"));
#======================== model create ======================================


#======================== reinforce learning ================================
for episode in range(1000):
    game.reset();
    # focusing to center of cart-pole so calculate position of cart-pole in image
    image_focus=(game.get_cart_location(game.image_width),int(game.image_height*0.6));
    image_cut_width = int(image_width*0.6);
    image_cut_height = int(image_height*0.4);

    #get current image from game
    now_screen = game.get_screen();

    #the image cut and resize
    now_screen=game.focus_cut_image(src=now_screen,focus=image_focus,width=image_cut_width,height=image_cut_height);
    now_screen=game.resize_image(now_screen,90,40);# get now state for image
    
    # define different_screen
    privous_screen = now_screen;
    different_screen = now_screen-privous_screen;#calculate different screen

    # the different_screen-image normalization and transform from HWC to CHW 
    float_dsc=np.array(different_screen,dtype=np.float32)/255.0;# RGB image(0~255) -> RGB image(0.0~1.0)
    now_state=float_dsc.transpose((2,0,1));# (Height, Width, Channel)->(Channel, Height, Width)
    
    score = 0
    for t  in range(1000):

        # Decide action from policy network
        action=RL.get_policy_action(state=now_state,action_num=2);

        # Execute action in Game environment by network policy
        observation,reward,done,info = game.set_control(action);
        
        #!!get current image after action from Game environment!!
        reward = reward if not done or score >= 499.0 else -100.0
        score += reward

        # Track down cart-pole
        image_focus=(game.get_cart_location(game.image_width),int(game.image_height*0.6));
        
        # get current image after action from game
        now_screen = game.get_screen();

        # image transform
        now_screen=game.focus_cut_image(src=now_screen,focus=image_focus,width=image_cut_width,height=image_cut_height);
        now_screen=game.resize_image(now_screen,90,40);# get now state for image
        
        different_screen = now_screen-privous_screen;#calculate different screen
        float_dsc=np.array(different_screen,dtype=np.float32)/255.0;# RGB image(0~255) -> RGB image(0.0~1.0)

        if not done:
            next_state = float_dsc.transpose((2,0,1));# (Height, Width, Channel)->(Channel, Height, Width)
        else:
            next_state = None;

        # stack results to memory
        RL.stack_memory(now_state,action,next_state,reward);
        
        # change from now data to previous data by time flow.
        now_state = next_state;
        privous_screen = now_screen;
        
        if(done==True):
            #print(observation)
            print("episode: ",episode,", done :",t)
            if(t>100):
                print("==================100=======")
            break;
         
        # update policy model
        RL.update(GAMMA=0.999,parameter_clamp=(-1,1));

    # target model synchronization with policy model.
    if episode%10==0:
        RL.target_update();

#======================== reinforce learning ================================

game.close();

