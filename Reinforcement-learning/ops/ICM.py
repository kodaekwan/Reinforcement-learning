import torch


class Fowrard_Model(torch.nn.Module):
    def __init__(self,action_dim,pi_dim,output_dim):
        super(Fowrard_Model,self).__init__();

        self.fc1 = torch.nn.Linear(pi_dim,256);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(action_dim,256);
        self.ac2 = torch.nn.ReLU();
        
        self.fc3 = torch.nn.Linear(256*2,512);
        self.ac3 = torch.nn.ReLU();

        self.fc4 = torch.nn.Linear(512,output_dim);

    def forward(self,action,pi):

        x1 =    self.ac1(
                self.fc1(pi));
        
        x2 =    self.ac2(
                self.fc2(action));
        
        concat_feature = torch.cat((x1,x2),-1);

        x=self.ac3(self.fc3(concat_feature));
        output=self.fc4(x);
        return output;


class Inverse_Model(torch.nn.Module):
    def __init__(self,pi_dim,action_dim):
        super(Inverse_Model,self).__init__();

        self.fc1 = torch.nn.Linear(pi_dim,128);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(pi_dim,128);
        self.ac2 = torch.nn.ReLU();

        self.fc3 = torch.nn.Linear(128*2,256);
        self.ac3 = torch.nn.ReLU();

        self.fc4 = torch.nn.Linear(256,action_dim);

    def forward(self,pi,next_pi):

        x1=self.ac1(self.fc1(pi));

        x2=self.ac2(self.fc2(next_pi));

        concat_feature = torch.cat((x1,x2),-1);

        x=self.ac3(self.fc3(concat_feature));
        output = self.fc4(x);
        return output;
        


class ICM_Model(torch.nn.Module):
    def __init__(self,state_dim,action_dim):
        super(ICM_Model,self).__init__();
        pi_dim = 128;
        
        #==================== state module ====================
        self.state_fc1 = torch.nn.Linear(state_dim,128);
        self.state_ac1 = torch.nn.ReLU();
        self.state_fc2 = torch.nn.Linear(128,pi_dim);
        self.state_ac2 = torch.nn.ReLU();
        #==================== state module ====================

        self.forward_model=Fowrard_Model(   action_dim=action_dim,
                                            pi_dim=pi_dim,
                                            output_dim=pi_dim);

        self.inverse_model=Inverse_Model(   pi_dim=pi_dim,
                                            action_dim=action_dim);
        
        self.pi_loss = torch.nn.MSELoss(reduction='none');

        self.inverse_loss = torch.nn.SmoothL1Loss();
        self.forward_loss = torch.nn.MSELoss();

    def forward(self,state,next_state,action):

        pi =        self.state_ac2(
                    self.state_fc2(
                    self.state_ac1(
                    self.state_fc1(state))));
        
        next_pi =   self.state_ac2(
                    self.state_fc2(
                    self.state_ac1(
                    self.state_fc1(next_state))));
        
        predic_action = self.inverse_model(pi,next_pi);
        predic_next_pi = self.forward_model(action,pi);

        return next_pi,predic_next_pi,predic_action;
    
    def get_intrinsic_reward_numpy(self,next_pi,predic_next_pi,eta=0.01):
        # eta is scaling factor
        intrinsic_reward = eta * self.pi_loss(next_pi,predic_next_pi).mean(-1)
        return intrinsic_reward.data.cpu().numpy();
    
    def get_intrinsic_reward(self,next_pi,predic_next_pi,eta=0.01):
        # eta is scaling factor
        intrinsic_reward = eta * self.pi_loss(next_pi,predic_next_pi).mean(-1)
        return intrinsic_reward;

    def get_inverse_loss(self,predic_action,real_action):
        loss=self.inverse_loss(predic_action,real_action);
        return loss;

    def get_forward_loss(self,predic_next_pi,next_pi):
        return self.forward_loss(predic_next_pi,next_pi);


#reference code:  https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/e8448777325493dd86f2c4164e7188882fc268ea/model.py#L148