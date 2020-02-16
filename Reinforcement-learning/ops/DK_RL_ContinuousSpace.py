from ops.DK_util import *
from ops import ICM





class DDPG_ICM_Module():
    # Deep Deterministic Policy Gradients
    def __init__(   self,
                    actor_net,
                    target_actor_net,
                    critic_net,
                    target_critic_net,
                    device=None,
                    batch_size=128,
                    train_start=0):

        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.icm = ICM.ICM_Model(state_dim=3,action_dim=1).to(self.device);

        self.batch_size = batch_size;
        self.train_start =train_start;
        self.actor_net=actor_net.to(self.device);
        self.target_actor_net=target_actor_net.to(self.device);
        self.critic_net=critic_net.to(self.device);
        self.target_critic_net=target_critic_net.to(self.device);
        
        self.actor_net.eval();
        self.target_actor_net.eval();
        self.critic_net.eval();
        self.target_critic_net.eval();

        hard_update(self.target_actor_net,self.actor_net);
        hard_update(self.target_critic_net,self.critic_net);


    def set_Noise(self,action_dim,action_limit,mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim;
        self.action_limit = action_limit;
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim,mu,theta,sigma);
    
    def set_ActorOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters());
        else:
            self.actor_optimizer = optimizer;
    
    def set_CriticOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.critic_optimizer = torch.optim.Adam(list(self.critic_net.parameters()) + list(self.icm.parameters()));
        else:
            self.critic_optimizer = optimizer;
    
    def set_Memory(self,capacity=5000,buffer_device=None):
        self.memory = ReplayMemory(capacity);
        if(buffer_device==None):
            self.buffer_device=self.device;
        else:
            self.buffer_device=buffer_device;
    
    def set_Criterion(self,criterion=None):    
        if(criterion==None):
            self.criterion = torch.nn.SmoothL1Loss().to(self.device);
        else:
            self.criterion = criterion.to(self.device);

    def get_exploitation_action(self,state):
        state = torch.autograd.Variable(torch.from_numpy(state)).to(self.device);
        action = self.target_actor_net.forward(state).detach();
        return action.item();
    
    
    def get_exploration_action(self,state):# add noise
        state = torch.autograd.Variable(torch.from_numpy(state)).to(self.device);
        action = self.actor_net.forward(state).detach();
        noise_action = action.item() + (self.noise.sample()*self.action_limit);
        return noise_action;
    
    def stack_memory(self,state=None,action=None,next_state=None,reward=None):
        # "state" type numpy
        # "action" type numpy
        # "next_state" type numpy
        # "reward" type numpy
        if (state is None) or (action is None) or (next_state is None) or (reward is None):
            return;
        
        self.memory.push(   torch.from_numpy(np.array(state)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(action)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(next_state)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(reward)).float().view(1,-1).to(self.buffer_device));

    def update(self,GAMMA = 0.99,soft_gain = 0.001):
        
        if(len(self.memory)<(self.batch_size+self.train_start)):
            return;

        self.actor_net.train();
        self.critic_net.train();
        
        batch_data = Transition(*zip(*self.memory.sample(self.batch_size)));

        state=torch.cat(batch_data.state).to(self.device);
        action=torch.cat(batch_data.action).to(self.device);
        reward=torch.cat(batch_data.reward).to(self.device);
        next_state=torch.cat(batch_data.next_state).to(self.device);


        # ============================== optimize icm ========================================
        beta = 0.5
        next_pi,predic_next_pi,predic_action = self.icm.forward(state,next_state,action);
        inverse_loss = self.icm.get_inverse_loss(predic_action,action);
        forward_loss = self.icm.get_forward_loss(predic_next_pi,next_pi.detach());
        with torch.no_grad():
            intrinsic_reward = self.icm.get_intrinsic_reward(next_pi,predic_next_pi,eta=0.01).unsqueeze(1);
        # ============================== optimize icm ========================================
        
        # ============================== optimize critic ========================================
        #       <double DQN>    Double DQN loss : reward + gamma*Qt(s', Qp(s',a') ) - Qp(s,a);
        #                       where  s == state, a == action, s' == next_state, a' == next_action
        #                       Qp is policy Q network, Qt is target Q network
        #
        #       <DDPG>          DDPG Critic loss : reward + gamma*Qtc(s', Qta(s',a') ) - Qpc(s,a);
        #                       Qpc is policy critic Q network, Qtc is target critic Q network
        #                       Qta is target action Q network

        self.critic_net.zero_grad();
        next_action =self.target_actor_net.forward(next_state).detach();
        next_val = self.target_critic_net.forward(next_state,next_action).detach();
        y_expected  = (reward+intrinsic_reward) + (GAMMA*next_val);# r + gamma*Qt'( s', Qp(s',a'))
        y_predicted = self.critic_net.forward(state,action); # Qp(s,a);
        loss_critic = self.criterion(y_predicted,y_expected);# r+gamma*Qt(s',a')-Qp(s,a);
        loss_critic += (((1.0-beta)*inverse_loss)+(beta*forward_loss));
        self.critic_optimizer.zero_grad();
        loss_critic.backward();
        self.critic_optimizer.step();
        # ============================== optimize critic ========================================



        

        # ============================== optimize action ========================================
        #       <DDPG>          DDPG Action loss : -Qpc(s,Qpa(s)) == -Qpc(s,a')
        self.actor_net.zero_grad();
        next_action_p = self.actor_net.forward(state);
        loss_actor = -self.critic_net.forward(state,next_action_p);
        self.actor_optimizer.zero_grad();
        loss_actor = loss_actor.mean();
        loss_actor.backward();
        self.actor_optimizer.step();
        # ============================== optimize action ========================================
        self.actor_net.eval();
        self.critic_net.eval();
        
        soft_update(self.target_actor_net,self.actor_net,soft_gain);
        soft_update(self.target_critic_net,self.critic_net,soft_gain);








class DDPG_Module():
    # Deep Deterministic Policy Gradients
    def __init__(   self,
                    actor_net,
                    target_actor_net,
                    critic_net,
                    target_critic_net,
                    device=None,
                    batch_size=128,
                    train_start=0):

        if(device==None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        else:
            self.device=device;
        
        self.batch_size = batch_size;
        self.train_start =train_start;
        self.actor_net=actor_net.to(self.device);
        self.target_actor_net=target_actor_net.to(self.device);
        self.critic_net=critic_net.to(self.device);
        self.target_critic_net=target_critic_net.to(self.device);
        
        self.actor_net.eval();
        self.target_actor_net.eval();
        self.critic_net.eval();
        self.target_critic_net.eval();

        hard_update(self.target_actor_net,self.actor_net);
        hard_update(self.target_critic_net,self.critic_net);


    def set_Noise(self,action_dim,action_limit,mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim;
        self.action_limit = action_limit;
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim,mu,theta,sigma);
    
    def set_ActorOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters());
        else:
            self.actor_optimizer = optimizer;
    
    def set_CriticOptimizer(self,optimizer=None):
        if(optimizer==None):
            self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters());
        else:
            self.critic_optimizer = optimizer;
    
    def set_Memory(self,capacity=5000,buffer_device=None):
        self.memory = ReplayMemory(capacity);
        if(buffer_device==None):
            self.buffer_device=self.device;
        else:
            self.buffer_device=buffer_device;
    
    def set_Criterion(self,criterion=None):    
        if(criterion==None):
            self.criterion = torch.nn.SmoothL1Loss().to(self.device);
        else:
            self.criterion = criterion.to(self.device);

    def get_exploitation_action(self,state):
        state = torch.autograd.Variable(torch.from_numpy(state)).to(self.device);
        action = self.target_actor_net.forward(state).detach();
        return action.item();
    
    
    def get_exploration_action(self,state):# add noise
        state = torch.autograd.Variable(torch.from_numpy(state)).to(self.device);
        action = self.actor_net.forward(state).detach();
        noise_action = action.item() + (self.noise.sample()*self.action_limit);
        return noise_action;
    
    def stack_memory(self,state=None,action=None,next_state=None,reward=None):
        # "state" type numpy
        # "action" type numpy
        # "next_state" type numpy
        # "reward" type numpy
        if (state is None) or (action is None) or (next_state is None) or (reward is None):
            return;
        
        self.memory.push(   torch.from_numpy(np.array(state)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(action)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(next_state)).float().view(1,-1).to(self.buffer_device),
                            torch.from_numpy(np.array(reward)).float().view(1,-1).to(self.buffer_device));

    def update(self,GAMMA = 0.99,soft_gain = 0.001):
        
        if(len(self.memory)<(self.batch_size+self.train_start)):
            return;

        self.actor_net.train();
        self.critic_net.train();
        
        batch_data = Transition(*zip(*self.memory.sample(self.batch_size)));

        state=torch.cat(batch_data.state).to(self.device);
        action=torch.cat(batch_data.action).to(self.device);
        reward=torch.cat(batch_data.reward).to(self.device);
        next_state=torch.cat(batch_data.next_state).to(self.device);

        # ============================== optimize critic ========================================
        #       <double DQN>    Double DQN loss : reward + gamma*Qt(s', Qp(s',a') ) - Qp(s,a);
        #                       where  s == state, a == action, s' == next_state, a' == next_action
        #                       Qp is policy Q network, Qt is target Q network
        #
        #       <DDPG>          DDPG Critic loss : reward + gamma*Qtc(s', Qta(s',a') ) - Qpc(s,a);
        #                       Qpc is policy critic Q network, Qtc is target critic Q network
        #                       Qta is target action Q network

        self.critic_net.zero_grad();
        next_action =self.target_actor_net.forward(next_state).detach();
        next_val = self.target_critic_net.forward(next_state,next_action).detach();
        y_expected  = reward + GAMMA*next_val;# r + gamma*Qt'( s', Qp(s',a'))
        y_predicted = self.critic_net.forward(state,action); # Qp(s,a);
        loss_critic = self.criterion(y_predicted,y_expected);# r+gamma*Qt(s',a')-Qp(s,a);
        self.critic_optimizer.zero_grad();
        loss_critic.backward();
        self.critic_optimizer.step();
        # ============================== optimize critic ========================================



        # ============================== optimize action ========================================
        #       <DDPG>          DDPG Action loss : -Qpc(s,Qpa(s)) == -Qpc(s,a')
        self.actor_net.zero_grad();
        next_action_p = self.actor_net.forward(state);
        loss_actor = -self.critic_net.forward(state,next_action_p);
        self.actor_optimizer.zero_grad();
        loss_actor = loss_actor.mean();
        loss_actor.backward();
        self.actor_optimizer.step();
        # ============================== optimize action ========================================
        self.actor_net.eval();
        self.critic_net.eval();
        
        soft_update(self.target_actor_net,self.actor_net,soft_gain);
        soft_update(self.target_critic_net,self.critic_net,soft_gain);






    


