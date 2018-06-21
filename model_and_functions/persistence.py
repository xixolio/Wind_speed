# -*- coding: utf-8 -*-

import numpy as np
import os

def train_and_test(vmin, vmax, y, y_ts):
    
    predicted_vector = y[-24:]
    
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax - vmin) + vmin
    
    mae = np.mean(np.abs(predicted_vector - y_ts))
    mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    mse = np.mean((predicted_vector - y_ts)**2)
                        
    return mae, mape, mse
    

def write_results(path,name,mae,mape,mse):
    
    for i in range(10):
            
            my_file = path+str(i)+name
            
            if not os.path.exists(my_file):
                
                f = open(path + str(i) + name, "a")
                #f.write("lags time_steps dense_nodes lstm_nodes processed scales\
                #        epochs l2 learning_rate mean_mae mean_mape mean_mse \
                #        std mae std_mape std_mse \n")
            
            else:
                
                f = open(path + str(i) + name, "a")
                    
            mean_mae = str(np.mean(mae[i]))
            mean_mape = str(np.mean(mape[i]))
            mean_mse = str(np.mean(mse[i]))

            f.write('{} {} {}  \n'.format( mean_mae, mean_mape, mean_mse) )
            
            f.close()