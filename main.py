# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:07:18 2018

@author: iaaraya
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
sys.path.append('/user/i/iaraya/CIARP/Wind_speed/data/')
sys.path.append('/user/i/iaraya/CIARP/Wind_speed/model_and_functions/')

from data_processing import get_data
import simple_LSTM as sLSTM

import numpy as np


if __name__ == "__main__":
    
    model = sys.argv[1]
        
    path = sys.argv[2]
        
    file_name = sys.argv[3]
        
    if model == "simple_LSTM":
      
        runs = 5
        
        sets = 10
           
        layers, lag, time_steps, epochs, l2, learning_rate = sLSTM.get_params(4)
    
        params = [layers, lag, time_steps, epochs, l2, learning_rate]
        
        training_inputs, testing_inputs, training_outputs, testing_outputs,\
        vmins, vmaxs = get_data(path, file_name, time_steps, lag)
            
        mae = np.zeros((sets, runs))
        mape = np.zeros((sets, runs))
        mse = np.zeros((sets, runs))
        
        for i in range(sets):
            
            X = training_inputs[i]
            X_ts = testing_inputs[i]
            y = training_outputs[i]
            y_ts = testing_outputs[i]
            
            for j in range(runs):
                
                model = sLSTM.model(layers, lag, time_steps, l2, learning_rate)
                
                mae[i,j], mape[i,j], mse[i,j], model = sLSTM.train_and_test(model, time_steps, lag, \
                                                      epochs, vmins[i], vmaxs[i],     \
                                                      X, y, X_ts, y_ts)
                
                model_name = "simple_LSTM_test_set_" + str(i) + "_run_" + str(j) +\
                '_'.join(str(x) for x in params)
                model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
               
        path = "/user/i/iaraya/CIARP/Wind_speed/results/"
        write_file_name = "simple_LSTM_test_" + file_name[:-4] + ".txt"
                
        sLSTM.write_results(path, write_file_name, params, mae, mape, mse)
        
        
