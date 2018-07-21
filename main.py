# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:07:18 2018

@author: iaaraya
"""
import os
#os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
sys.path.append('/user/i/iaraya/Wind_speed/data/')
sys.path.append('/user/i/iaraya/Wind_speed/model_and_functions/')
#sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/data/')
#sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/model_and_functions/')
#sys.path.append('/home/iaraya/CIARP/Wind_speed/data/')
#sys.path.append('/home/iaraya/CIARP/Wind_speed/model_and_functions/')


from data_processing import get_data
import simple_LSTM as sLSTM
import hierarchical_LSTM as hLSTM
import persistence
import copy
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
                                                      X, y, copy.deepcopy(X_ts), copy.deepcopy(y_ts))
                
                model_name = "simple_LSTM_test_set_" + str(i) + "_run_" + str(j) +\
                '_'.join(str(x) for x in params)
                #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
               
        path = "/user/i/iaraya/Wind_speed/results/"
        write_file_name = "finalsimple_LSTM_" + file_name[:-4] + ".txt"
                
        sLSTM.write_results(path, write_file_name, params, mae, mse,runs)
        
        
    elif model == "hierarchical_LSTM":
      
        runs = 5
        
        sets = 10
           
        lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
        epochs, l2, batch_size, shuffle = hLSTM.get_params(4)
    
        params = [lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                   epochs, l2, batch_size, shuffle]
        
        training_inputs_sets = []
        testing_inputs_sets = []
        
        for i in range(len(lags)):
            
            training_inputs, testing_inputs, training_outputs, testing_outputs,\
            vmins, vmaxs = get_data(path, file_name, time_steps[i], lags[i],overlap=False)
            
            training_inputs_sets.append(training_inputs)
            testing_inputs_sets.append(testing_inputs)
            

            
        mae = np.zeros((sets, runs))
        mape = np.zeros((sets, runs))
        mse = np.zeros((sets, runs))
        
        for i in range(sets):
            
            min_data_len = 10000000
            
            for j in range(len(lags)):
                
                if len(training_inputs_sets[j][i]) < min_data_len:
                    
                    min_data_len = len(training_inputs_sets[j][i])
            
            X = []
            X_ts = []
                        
            for j in range(len(lags)):
                
                 X.append(training_inputs_sets[j][i][-min_data_len:])
                 X_ts.append(testing_inputs_sets[j][i])
    
            y = training_outputs[i][-min_data_len:]
            y_ts = testing_outputs[i]
            
            for j in range(runs):
                
                model = hLSTM.model(lags, time_steps, processed_scales, \
                                    dense_nodes, lstm_nodes, l2)
                
                mae[i,j], mape[i,j], mse[i,j], model = hLSTM.train_and_test(model, time_steps, lags, \
                                                      epochs, vmins[i], vmaxs[i],     \
                                                      X, y, copy.deepcopy(X_ts), y_ts, batch_size = batch_size, \
                                                      shuffle = shuffle)
                
                model_name = "hierarchical2_LSTM_set_" + str(i) + "_run_" + str(j) +\
                '_'.join(str(x) for x in params)
                #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
               
        path = "/user/i/iaraya/Wind_speed/results/"
        #path = "/home/iaraya/CIARP/Wind_speed/results/"
        write_file_name = "final2_hierarchical_LSTM_" + file_name[:-4] + ".txt"
                
        hLSTM.write_results(path, write_file_name, params, mae, mse,runs)
        
        
    elif model == "persistence":
        
        training_inputs, testing_inputs, training_outputs, testing_outputs,\
        vmins, vmaxs = get_data(path, file_name, 1, 1)
        
        mae = np.zeros((sets))
        mape = np.zeros((sets))
        mse = np.zeros((sets))
        
        for i in range(sets):
            
            X = training_inputs[i]
            X_ts = testing_inputs[i]
            y = training_outputs[i]
            y_ts = testing_outputs[i]
            
            #model = sLSTM.model(layers, lag, time_steps, l2, learning_rate)
            
            mae[i], mape[i], mse[i] = persistence.train_and_test(vmins[i], vmaxs[i], y, y_ts)
            
            model_name = "persistence_" + str(i) + "_run"
            #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
           
        path = "results/"
        write_file_name = "persistence_" + file_name[:-4] + ".txt"
                
        persistence.write_results(path, write_file_name, mae,mape, mse)
        
       