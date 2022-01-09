import numpy as np
import gym
import controller
import torch
env = gym.make('Pendulum-v1',g=0.0)
env.seed(1); 

model = torch.nn.Sequential(
    torch.nn.Linear(3, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 1),
    torch.nn.Tanh()
)

def convertState(state):
    cos_theta = state[0];
    sin_theta = state[1];
    thetadot = state[2];
    return np.arctan2(sin_theta,cos_theta),thetadot;

def get_random_index(x_data):
    index_buff = [];
    for i in range(len(x_data)):
        index_buff.append(i);
    np.random.shuffle(index_buff)
    return index_buff;


pid = controller.PID(0.3,0.5,0.1,output_limit=2.0);


target = 0.0;# 0 deg
theta = 0.0;# rad
thetadot = 0.0;# rad/s

p_error = 0.0;
i_error = 0.0;
d_error = 0.0;

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

train_output = torch.zeros([1,1]);
train_x_buffer = [];
train_y_buffer = [];

for eps in range(10):
    print("!!!Data Collect!!! "+str(eps)+" situation");
    error_k_0 = 0.0;
    error_k_1 = 0.0;
    error_k_2 = 0.0;
    state = env.reset();
    pid.reset();
    for time in range(500):
        # show Pendulum
        env.render();

        # convert The Pendulum-v1 state to angle and angle velocity
        theta,thetadot  = convertState(state);
        refer_angle = np.rad2deg(theta);

        action = pid.control(target,refer_angle,dt=0.05);
        state,reward,done,_ = env.step([action]);

        error_k_0 = target-refer_angle;
        # collect train data
        train_x_buffer.append(np.array([error_k_0,error_k_1,error_k_2],dtype=np.float32).copy());
        train_y_buffer.append(np.array([action],dtype=np.float32).copy());
        error_k_2 = error_k_1;
        error_k_1 = error_k_0;

        if done:
            break;

batch_train_x = [];
batch_train_y = [];
count = 0;
model.train();
for eps in range(100):
    rand_index = get_random_index(train_y_buffer);
    for index_ in rand_index:
        batch_train_x.append(train_x_buffer[index_]);
        batch_train_y.append(train_y_buffer[index_]);
        
        if(len(batch_train_y)>32):
            batch_train_x.pop(0)
            batch_train_y.pop(0)
        else:
            continue;

        error_k_0 = target-refer_angle;
        train_input=torch.from_numpy(np.array(batch_train_x,dtype=np.float32));
        train_output = model(train_input/180.0);
        control_output =torch.from_numpy(np.array(batch_train_y,dtype=np.float32));
        loss = criterion(train_output*2.0, control_output);
        if count % 100 == 99:
            print(count, loss.item());

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        count +=1;


model.eval();
for eps in range(10):
    target = 10.0*eps;# 0 deg
    print("!!!Test!!! => Target : "+str(target)+"deg");
    state = env.reset();
    error_k_0 = 0.0;
    error_k_1 = 0.0;
    error_k_2 = 0.0;
    
    for time in range(500):
        # show Pendulum
        env.render();
        theta,thetadot  = convertState(state);
        refer_angle = np.rad2deg(theta);

        error_k_0 = target-refer_angle;
        intput=torch.from_numpy(np.array([error_k_0,error_k_1,error_k_2],dtype=np.float32)).unsqueeze(0);
        action = model(intput/180.0);
        action = (action*2.0).squeeze(0).item();
        error_k_2 = error_k_1;
        error_k_1 = error_k_0;
        state,reward,done,_ = env.step([action]);


            
env.close();