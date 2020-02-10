import gym
import time
import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

class Episode_Threshold():#random Threshold
    def __init__(self,EPS_START = 0.9,EPS_END = 0.05,EPS_DECAY = 200):
        self.EPS_START=EPS_START;
        self.EPS_END=EPS_END;
        self.EPS_DECAY=EPS_DECAY;
        self.step=0;
    def get_threshold(self):
        eps_threshold = self.EPS_END+ (self.EPS_START-self.EPS_END)*math.exp(-1. * self.step / self.EPS_DECAY);
        self.step+=1;
        return eps_threshold;

class GAME():
    def __init__(self,game_name='CartPole-v0'):
        self.env = gym.make(game_name);# make game
        self.reset();
        
    def get_cart_location(self,image_width):
        # you can get cart position from pixel unit
        # env.state[0] is cart position
        world_width = self.env.x_threshold * 2;
        scale = image_width/world_width;
        return int(self.env.state[0]*scale + image_width/2.0);
    
    def reset(self):
        self.env.reset();
        image_shape = self.get_screen().shape;
        self.max_key_num = self.env.action_space.n;
        self.image_height = image_shape[0];
        self.image_width = image_shape[1];
        self.image_channel = image_shape[2];
        self.cart_location=self.get_cart_location(self.image_width);
    
    def get_screen(self):
        return self.env.render(mode='rgb_array');
    
    def set_control(self,key):
        return self.env.step(key);

    def close(self):
        self.env.close();
    
    def cut_image(self,src,x=0,y=0,width=600,height=400):
        #src (H,W,C)
        return src[y:y+height,x:x+width:];
    
    def focus_cut_image(self,src,focus=(0,1),width=600,height=400):
        # "src" is numpy type and have shape to (H,W,C)
        # "focus" is (x,y)
        # "width" is cutting range from "focus x"
        # "height" is cutting range from "focus y".
        screen_shape=src.shape;
        
        screen_height=screen_shape[0];
        screen_width=screen_shape[1];

        if focus[0] < width//2:
            w_slice_range = slice(0,width)
        elif focus[0] > (screen_width - (width//2) ):
            w_slice_range = slice(-width,None);
        else:
            w_slice_range = slice((focus[0]-(width//2)), (focus[0]+(width//2)));
        
        if focus[1] < height//2:
            h_slice_range = slice(0,height)
        elif focus[1] > (screen_height - (height//2) ):
            h_slice_range = slice(-height,None);
        else:
            h_slice_range = slice((focus[1]-(height//2)), (focus[1]+(height//2)));

        return src[h_slice_range,w_slice_range,:];
    
    def resize_image(self,src,width,height):
        return cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_CUBIC);

game=GAME();
image_height=game.image_height;
image_width=game.image_width;
print("image height : ",game.image_height);
print("image width : ",game.image_width);

def get_rand_key(max_key_num):
    # if max_key_num=2 , you can get 0 or 1
    # if max_key_num=3 , you can get 0 or 1 or 2
    return np.random.randint(max_key_num);

image_focus=(game.cart_location,int(game.image_height*0.6));
image_cut_width = int(image_width*0.6);
image_cut_height = int(image_height*0.4);

for _  in range(100):
    screen =game.get_screen();
    screen=game.focus_cut_image(src=screen,focus=image_focus,width=image_cut_width,height=image_cut_height);
    screen=game.resize_image(screen,90,40);

    print("screen shape : ",screen.shape)
    observation,reward,done,info = game.set_control(get_rand_key(game.max_key_num));

    screen2 =game.get_screen();
    screen2=game.focus_cut_image(src=screen2,focus=image_focus,width=image_cut_width,height=image_cut_height);
    screen2=game.resize_image(screen2,90,40);

    plt.imshow(screen2-screen,interpolation='none');
    plt.pause(0.01);
    if(done==True):
        
        print(observation)
        print("done")
        game.reset();
        image_focus=(game.cart_location,int(game.image_height*0.6));
        continue;

game.close();
plt.close();
exit();







#episode_threshold=Episode_Threshold(EPS_START=0.9,EPS_END=0.05,EPS_DECAY=200);



