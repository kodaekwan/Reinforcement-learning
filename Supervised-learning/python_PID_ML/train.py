import numpy as np
import gym
import controller
import torch
import traindatacollector
from matplotlib import pyplot as plt

env = gym.make('Pendulum-v1',g=0.0)
env.seed(1); 
datacollector = traindatacollector.DataCollect();

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

pid = controller.PID(0.3,0.6,0.1,output_limit=2.0);


target = 0.0;# 0 deg
theta = 0.0;# rad
thetadot = 0.0;# rad/s

for eps in range(10):
    print("!!!Data Collect!!! "+str(eps)+" situation");
    error_k_0 = 0.0;
    error_k_1 = 0.0;
    error_k_2 = 0.0;
    state = env.reset();
    pid.reset();
    for time in range(500):
        # show Pendulum
        if eps == 0:
            env.render();

        # convert The Pendulum-v1 state to angle and angle velocity
        theta,thetadot  = convertState(state);
        refer_angle = np.rad2deg(theta);

        action = pid.control(target,refer_angle,dt=0.05);
        state,reward,done,_ = env.step([action]);

        error_k_0 = target-refer_angle;
        # collect train data
        datacollector.collect([error_k_0,error_k_1,error_k_2],[action])
        error_k_2 = error_k_1;
        error_k_1 = error_k_0;

        if done:
            break;


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
datacollector.show();
count = 0;
model.train();

for eps in range(100):
    done = False;
    while(not done):
        batch_train_x,batch_train_y,done = datacollector.get_data(32);
        if done:
            break;

        train_input=torch.from_numpy(batch_train_x);
        train_output = model(train_input/360.0);
        control_output =torch.from_numpy(batch_train_y);
        loss = criterion(train_output*2.0, control_output);
        if count % 100 == 99:
            print(count, loss.item());
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        count +=1;

test_ref = [];
test_tar = [];
test_tor = [];

pid_ref = [];
pid_tar = [];
pid_tor = [];

model.eval();
for eps in range(1):
    
    
    error_k_0 = 0.0;
    error_k_1 = 0.0;
    error_k_2 = 0.0;    
    Hz = 1.0;
    dt = 0.05;
    state = env.reset();
    print("!!!ML Test!!! => Target : "+"sin");
    for time in range(500):
        target = 90.0*np.sin(Hz*time*dt);# 0 deg
        # show Pendulum
        env.render();
        theta,thetadot  = convertState(state);
        refer_angle = np.rad2deg(theta);

        error_k_0 = target-refer_angle;
        intput=torch.from_numpy(np.array([error_k_0,error_k_1,error_k_2],dtype=np.float32)).unsqueeze(0);
        action = model(intput/360.0);
        action = (action*2.0).squeeze(0).item();
        error_k_2 = error_k_1;
        error_k_1 = error_k_0;
        state,reward,done,_ = env.step([action]);

        test_ref.append(refer_angle);
        test_tar.append(target);
        test_tor.append(action);
    
    print("!!!PID Test!!! => Target : "+"sin");
    pid.reset();
    state = env.reset();
    for time in range(500):
        target = 90.0*np.sin(Hz*time*dt);# 0 deg
        # show Pendulum
        env.render();
        theta,thetadot  = convertState(state);
        refer_angle = np.rad2deg(theta);

        action = pid.control(target,refer_angle,dt=0.05);
        state,reward,done,_ = env.step([action]);

        pid_ref.append(refer_angle);
        pid_tar.append(target);
        pid_tor.append(action);

plt.subplot(5,1,1);
plt.title('ML')
plt.plot(test_ref,label='ref');
plt.plot(test_tar,label='tar');
plt.legend(loc='best')

plt.subplot(5,1,3);
plt.title('PID')
plt.plot(pid_ref,label='ref');
plt.plot(pid_tar,label='tar');
plt.legend(loc='best')

plt.subplot(5,1,5);
plt.title('ML vs PID')
plt.plot(test_tor,label='ML-output');
plt.plot(pid_tor,label='PID-output');
plt.legend(loc='best')
plt.savefig("Supervised-learning/python_PID_ML/result.png")  
plt.show();
ML_loss = np.mean(np.abs(np.array(test_ref)-np.array(test_tar))[100:])
PID_loss = np.mean(np.abs(np.array(pid_ref)-np.array(pid_tar))[100:])

print("ML loss: "+str(ML_loss)+", PID loss : "+str(PID_loss));

env.close();

# reference paper : On Replacing PID Controller with Deep Learning Controller for DC Motor System 
# DOI:10.12720/joace.3.6.452-456