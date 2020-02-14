import sys
sys.path.append("..")
import ops.DK_RL_DiscreteSpace as DiscreteSpace



class GAME():
    import cv2
    def __init__(self,game_name='CartPole-v0'):
        import gym #$ pip3 install gym
        self.game_name=game_name
        self.env = gym.make(self.game_name);# make game
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
        
        if self.game_name == 'Pendulum-v0':
            self.max_key_num = 1;
        else:
            self.max_key_num = self.env.action_space.n;
            
            
        self.image_height = image_shape[0];
        self.image_width = image_shape[1];
        self.image_channel = image_shape[2];

        if self.game_name == 'CartPole-v0' or self.game_name == 'CartPole-v1':
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
        return self.cv2.resize(src, dsize=(width, height), interpolation=self.cv2.INTER_CUBIC);
