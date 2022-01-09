import numpy as np


class PID:
    def __init__(self,kp,ki,kd,output_limit=1.0):
        self.kp = kp#proportional gain
        self.ki = ki#integral gain
        self.kd = kd#derivative gain
        self.output_limit = output_limit
        if(ki != 0.0):
            self.buffer_limit = output_limit/ki;
        self.reset();
    def reset(self):
        self.previous_error = 0.0
        self.intergal_buffer = 0.0;

    def control(self,target,refer,dt=0.001):
        error = target-refer
        integral = self.intergal_buffer + (error*dt);
        if(self.ki != 0.0):
            integral = np.clip(integral,-self.buffer_limit,self.buffer_limit )
        derivative = (error - self.previous_error)/dt;
        
        output = (self.kp*error) + (self.ki*integral) + (self.kd*derivative);
        
        self.intergal_buffer = integral;
        self.previous_error = error;

        return np.clip(output,-self.output_limit,self.output_limit);

