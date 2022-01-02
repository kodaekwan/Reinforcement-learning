import numpy as np
import gym
import PID_Model
env = gym.make('Pendulum-v1')
env.seed(1); 

def convertState(state):
    cos_theta = state[0];
    sin_theta = state[1];
    thetadot = state[2];
    return np.arctan2(sin_theta,cos_theta),thetadot;

state = env.reset();
controller = PID_Model.PID(0.3,0.5,0.1,output_limit=2.0);

target = 0.0;# 0 deg
theta = 0.0;# rad
thetadot = 0.0;# rad/s

for time in range(500):
    env.render();
    #action = env.action_space.sample()
    
    theta,thetadot  = convertState(state);
    refer_angle = np.rad2deg(theta);

    if(np.abs(refer_angle)<15.0):
        action = controller.control(target,refer_angle,dt=0.05);
    else:
        if (thetadot == 0.0):# if velocity of pendulum is zero
            action = 2.0;
        else:
            action = 1.0/thetadot;

    state,reward,done,_ = env.step([action]);

    if done:
        break;
        
env.close();