import numpy as np


class DataCollect:
    def __init__(self):
        self.reset();
        pass;

    def reset(self):
        self.train_x_buffer = [];
        self.train_y_buffer = [];
        self.rand_index = [];
        self.get_count=0;
    

    def collect(self,input_x,groundtruth):
        self.train_x_buffer.append(np.array(input_x,dtype=np.float32).copy());
        self.train_y_buffer.append(np.array(groundtruth,dtype=np.float32).copy());
    
    def show(self):
        print("train x : "+str(np.array(self.train_x_buffer).shape)+", train y : "+str(np.array(self.train_y_buffer).shape));

    def get_random_index(self):
        buff = [];
        for i in range(len(self.train_y_buffer)):
            buff.append(i);
        np.random.shuffle(buff)
        return buff;

    def get_data(self,batch_size = 32):
        if(self.get_count == 0):
            self.rand_index = self.get_random_index()
      
        if(self.get_count>=int(len(self.train_x_buffer)/batch_size)):
            self.rand_index = [];
            self.get_count = 0;
            return None, None, True;

        batch_x = [];
        batch_y = [];
        if (len(self.rand_index)<batch_size):
            for index_ in self.rand_index:
                batch_x.append(self.train_x_buffer[index_]);
                batch_y.append(self.train_y_buffer[index_]);
        else:
            for num in range(batch_size):
                index_=self.rand_index[num+(batch_size*self.get_count)];
                batch_x.append(self.train_x_buffer[index_]);
                batch_y.append(self.train_y_buffer[index_]);
            
        self.get_count+=1;
        return np.array(batch_x,dtype=np.float32), np.array(batch_y,dtype=np.float32), False;

            
        




