import gym # $ pip3 install gym
import time
import random
import numpy as np # $ pip3 install numpy

env = gym.make('CartPole-v0');# install game
observation = env.reset();# game environment reset
print(observation);# print initial observation, observation are cart joint angle and joint speed value.

max_keynum = env.action_space.n;# you can get to steerable key-number.
print("cotrol key num:",max_keynum);# you can see to steerable key-number.

image_shape = env.render(mode='rgb_array').shape;
print("image shape : ",image_shape);# print image shape

image_height = image_shape[0];
image_width = image_shape[1];
image_channel = image_shape[2];

def get_rand_key(max_key_num):
    # if max_key_num=2 , you can get 0 or 1
    # if max_key_num=3 , you can get 0 or 1 or 2
    return np.random.randint(max_key_num);

for _  in range(100):
    screen =env.render(mode='rgb_array');# get game screen image by RGB
    observation,reward,done,info = env.step(get_rand_key(max_keynum)); #env.step() should give parameter by control value(0 or 1)
    time.sleep(0.1);# slow view
    if(done==True):
        print("done")
        break;
env.close();