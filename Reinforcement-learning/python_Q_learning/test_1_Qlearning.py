import numpy as np 
import matplotlib.pyplot as plt
import time


class Mygame:
    # 직접만든 게임 ㅋㅋ
    def __init__(self):
        self.env=np.zeros((6,6),dtype=np.int32);
        for i in range(6):
            self.env[0,i]=-1;
            self.env[i,0]=-1;
            self.env[5,i]=-1;
            self.env[i,5]=-1;
        self.env[2,2]=-1;
        self.env[2,4]=-1;
        self.env[3,4]=-1;
        self.env[4,1]=-1;
        
        self.env[4,4]= 1;
        self.now_state = [1,1];

    def reset(self):
        self.now_state=[1,1];
        return ((self.now_state[0]-1)*4)+(self.now_state[1]-1);
    
    def render(self):
        env=self.env.copy();
        env[self.now_state[0],self.now_state[1]]=9;
        print(env);

    def step(self,action):
        if(action==0):
            self.now_state[0]-=1;
        elif(action==1):
            self.now_state[1]+=1;
        elif(action==2):
            self.now_state[0]+=1;
        elif(action==3):
            self.now_state[1]-=1;
        
        reward = self.env[self.now_state[0],self.now_state[1]];
        state = ((self.now_state[0]-1)*4)+(self.now_state[1]-1);

        if(reward==-1):
            done = True;
            reward=0;
            if(action==0):
                self.now_state[0]+=1;
            elif(action==1):
                self.now_state[1]-=1;
            elif(action==2):
                self.now_state[0]-=1;
            elif(action==3):
                self.now_state[1]+=1;
            
            state = ((self.now_state[0]-1)*4)+(self.now_state[1]-1);

        elif(reward==1):
            done = True;
            reward=1;
        else:
            done = False;
            reward=0;


        return state,reward,done;


def Q_Frozen():

    space_size = 16;# "FrozenLake-v0"라는 게임의 크기
    action_size = 4;# "FrozenLake-v0"라는 게임의 움직임에 대한 권한수 ex) [up,down,left,right] => env.action_space.n= 4

    num_episodes = 2000;

    rList = [];
    gg=Mygame();
    Q = np.zeros((space_size,action_size));

    for i in range(num_episodes):
        
        #now_state = env.reset(); # 게임 환경 초기화 => 초기 상태로 맞춤. ex)now_state = env.reset() = 0
        now_state=gg.reset();
        next_state=0;
        rAll = 0;
        done = False;
        e = 1.0;
        discount_reward = 0.99;
        np.random.seed();
        while not done:
            action = np.argmax( Q[now_state,:] + (np.random.randn(1,4)/((i/1000)+1)) );

            next_state, reward, done = gg.step(action);
            if(done==True and reward!=1):
                break;
            
            Q[now_state,action] = reward +discount_reward*np.amax(Q[next_state,:]);

            rAll +=reward
            now_state = next_state;          
            
     
        rList.append(rAll);
    
    
    print(Q);
    print("success rate:" + str(sum(rList)/num_episodes ));
    plt.bar(range(len(rList)),rList,color='blue');
    plt.show();
    
    now_state=gg.reset();
    done = False;
    gg.render();
    while not done:
        
        action = np.argmax(Q[now_state,:]);
        now_state, reward, done = gg.step(action);
        gg.render();
        time.sleep(1);





Q_Frozen();