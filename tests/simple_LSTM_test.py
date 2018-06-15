# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:07:18 2018

@author: iaaraya
"""

import sys
#sys.path.append('../')
sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/data/')
sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/model_and_functions/')

#from data_processing.data_processing import get_data
from data_processing import get_data

#import model_and_functions.simple_LSTM as sLSTM
import simple_LSTM as sLSTM
import numpy as np


if __name__ == "__main__":
    
    path = sys.argv[1]
    
    file_name = sys.argv[2]
    
    print("Printing path and file_name")
    print("path: " + path)
    print("file_name: " + file_name)
    
    runs = 5
    
    sets = 10
       
    layers, lag, time_steps, epochs, l2, learning_rate = sLSTM.get_params(3)
    
    print("Printing parameters")
    print("layers: " + str(layers))
    print("lag: " + str(lag))
    print("time_steps: " + str(time_steps))
    print("epochs: " + str(epochs))
    print("l2: " + str(l2))
    print("learning_rate: " + str(learning_rate))

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
        
        print("Printing data shapes")
        print("X shape: " + str(X.shape))
        print("X_ts shape: " + str(X_ts.shape))
        print("y shape: " + str(y.shape))
        print("y_ts shape: " + str(y_ts.shape))
        print("vmins: " + str(vmins))
        print("vmaxs: " + str(vmaxs))
    
        print("Reducing data_size")
        
        #X = X[-100:,:]
        #y = y[-100:,:]
        
        for j in range(runs):
            
            model = sLSTM.model(layers, lag, time_steps, l2, learning_rate)
            
            print("Printing model layers")
            print(model.layers)
            
            mae[i,j], mape[i,j], mse[i,j], model = sLSTM.train_and_test(model, time_steps, lag, \
                                                  epochs, vmins[i], vmaxs[i],     \
                                                  X, y, X_ts, y_ts)
            
            model_name = "simple_LSTM_test_set_" + str(i) + "_run_" + str(j) +\
            '_'.join(str(x) for x in params)
            model.save("../models/" + model_name + ".h5")
            
    print("Printing errors")
    print("Mae mape mse :" + str(mae) + str(mape) + str(mse))
            
    write_file_name = "simple_LSTM_test" + file_name[-4:] + ".txt"
            
    sLSTM.write_results(path, write_file_name, params, mae, mape, mse)
        
        
