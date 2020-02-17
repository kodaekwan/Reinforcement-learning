import torch


class Fowrard_Model(torch.nn.Module):
    def __init__(self,action_dim,phi_dim,output_dim):
        super(Fowrard_Model,self).__init__();

        self.fc1 = torch.nn.Linear(phi_dim,256);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(action_dim,256);
        self.ac2 = torch.nn.ReLU();
        
        self.fc3 = torch.nn.Linear(256*2,512);
        self.ac3 = torch.nn.ReLU();

        self.fc4 = torch.nn.Linear(512,output_dim);

    def forward(self,action,phi):

        x1 =    self.ac1(
                self.fc1(phi));
        
        x2 =    self.ac2(
                self.fc2(action));
        
        concat_feature = torch.cat((x1,x2),-1);

        x=self.ac3(self.fc3(concat_feature));
        output=self.fc4(x);
        return output;


class Inverse_Model(torch.nn.Module):
    def __init__(self,phi_dim,action_dim):
        super(Inverse_Model,self).__init__();

        self.fc1 = torch.nn.Linear(phi_dim,128);
        self.ac1 = torch.nn.ReLU();

        self.fc2 = torch.nn.Linear(phi_dim,128);
        self.ac2 = torch.nn.ReLU();

        self.fc3 = torch.nn.Linear(128*2,256);
        self.ac3 = torch.nn.ReLU();

        self.fc4 = torch.nn.Linear(256,action_dim);

    def forward(self,phi,next_phi):

        x1=self.ac1(self.fc1(phi));

        x2=self.ac2(self.fc2(next_phi));

        concat_feature = torch.cat((x1,x2),-1);

        x=self.ac3(self.fc3(concat_feature));
        output = self.fc4(x);
        return output;
        


class ICM_Model(torch.nn.Module):
    def __init__(self,state_dim,action_dim,phi_dim = 128):
        super(ICM_Model,self).__init__();
        
        
        #==================== state module ====================
        self.state_fc1 = torch.nn.Linear(state_dim,128);
        self.state_ac1 = torch.nn.ReLU();
        self.state_fc2 = torch.nn.Linear(128,phi_dim);
        self.state_ac2 = torch.nn.ReLU();
        #==================== state module ====================

        self.forward_model=Fowrard_Model(   action_dim=action_dim,
                                            phi_dim=phi_dim,
                                            output_dim=phi_dim);

        self.inverse_model=Inverse_Model(   phi_dim=phi_dim,
                                            action_dim=action_dim);
        
        self.intrinsic_reward_distance = torch.nn.MSELoss(reduction='none');

        self.inverse_loss = torch.nn.SmoothL1Loss();
        self.forward_loss = torch.nn.MSELoss();

    def forward(self,state,next_state,action):

        phi =       self.state_ac2(
                    self.state_fc2(
                    self.state_ac1(
                    self.state_fc1(state))));
        
        next_phi =  self.state_ac2(
                    self.state_fc2(
                    self.state_ac1(
                    self.state_fc1(next_state))));
        
        predic_action = self.inverse_model(phi,next_phi);
        predic_next_phi = self.forward_model(action,phi);

        return next_phi,predic_next_phi,predic_action;
    
    def get_intrinsic_reward_numpy(self,next_phi,predic_next_phi,eta=0.01):
        # eta is scaling factor
        intrinsic_reward = eta * self.intrinsic_reward_distance(next_phi,predic_next_phi).mean(-1)
        return intrinsic_reward.data.cpu().numpy();
    
    def get_intrinsic_reward(self,next_phi,predic_next_phi,eta=0.01):
        # eta is scaling factor
        intrinsic_reward = eta * self.intrinsic_reward_distance(next_phi,predic_next_phi).mean(-1)
        return intrinsic_reward;

    def get_inverse_loss(self,predic_action,action):
        loss=self.inverse_loss(predic_action,action);
        return loss;

    def get_forward_loss(self,predic_next_phi,next_phi):
        return self.forward_loss(predic_next_phi,next_phi);


#reference code:  https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/e8448777325493dd86f2c4164e7188882fc268ea/model.py#L148