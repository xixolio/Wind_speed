# -*- coding: utf-8 -*-

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

sys.path.append('/home/iaraya/Wind_speed/data/')
sys.path.append('/home/iaraya/Wind_speed/model_and_functions/')

from data_processing import get_data
import simple_LSTM as sLSTM
import hierarchical_LSTM as hLSTM
import persistence
import keras.backend as K
import numpy as np


if __name__ == "__main__":
    
    model = sys.argv[1]
        
    path = sys.argv[2]
        
    file_name = sys.argv[3]
    
    data_path = path + "data/"
    
    results_path = path + "results/"
        
    if model == "simple_LSTM":
      
        runs = 3
        
        sets = 2
        
        counter = 0
        
        parameters_set = sLSTM.get_params_gpu(4)
        
        for params in parameters_set:
            
            print(counter)
            
            counter += 1
           
            [layers, lag, time_steps, epochs, l2, learning_rate, batch_size, verbose] = params
            
            training_inputs, testing_inputs, training_outputs, testing_outputs,\
            vmins, vmaxs = get_data(data_path, file_name, time_steps, lag)
                
    #        mae = np.zeros((sets, runs))
    #        mape = np.zeros((sets, runs))
    #        mse = np.zeros((sets, runs))
            
            for i in range(sets):
                
                X = training_inputs
                X_ts = testing_inputs
                y = training_outputs
                y_ts = testing_outputs
                
            
                    
            model = sLSTM.model_gpu(layers, lag, time_steps, l2, learning_rate, sets, runs)
            
            mae, mape, mse, model = sLSTM.train_and_test_gpu(model, time_steps, lag, \
                                                  epochs, vmins, vmaxs,     \
                                                  X, y, X_ts, y_ts, sets, runs, batch_size,
                                                  verbose)
            
            #model_name = "simple_LSTM_test_set_" + str(i) + "_run_" + str(j) +\
            #'_'.join(str(x) for x in params)
            
            #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
                   
            #path = "/user/i/iaraya/CIARP/Wind_speed/results/"
            
            write_file_name = "simple_LSTM_test_" + file_name[:-4] + ".txt"
                    
            sLSTM.write_results(results_path, write_file_name, params, mae, mape, mse)
        
        
    elif model == "hierarchical_LSTM":
      
        runs = 3
        
        sets = 10
           
        parameters_set = hLSTM.get_params_gpu(4)
        
        counter = 0
        
        for params in parameters_set:
            
            print(counter)
            
            counter += 1
        
            lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
            epochs, l2, batch_size, shuffle, verbose = params
            
            training_inputs_sets = []
            testing_inputs_sets = []
            
            for i in range(len(lags)):
                
                training_inputs, testing_inputs, training_outputs, testing_outputs,\
                vmins, vmaxs = get_data(data_path, file_name, time_steps[i], lags[i],overlap=False)
                
                training_inputs_sets.append(training_inputs)
                testing_inputs_sets.append(testing_inputs)
                               
            mae = np.zeros((sets, runs))
            mape = np.zeros((sets, runs))
            mse = np.zeros((sets, runs))
            
            X_sets = []
            X_ts_sets = []
            y_sets = []
            y_ts_sets = []
            
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
                
                X_sets.append(X)
                X_ts_sets.append(X_ts)
                y_sets.append(y)
                y_ts_sets.append(y_ts)
                
            # Model creation and training
                    
            model = hLSTM.model_gpu(lags, time_steps, processed_scales, \
                                dense_nodes, lstm_nodes, l2, runs, sets)
            
            #print("FIRST first like")
            #print(X_ts_sets)
            mae, mape, mse, model = hLSTM.train_and_test_gpu(model, time_steps, lags, \
                                                  epochs, vmins, vmaxs,     \
                                                  X_sets, y_sets, X_ts_sets, y_ts_sets,sets, runs, verbose = verbose,\
                                                  batch_size = batch_size, shuffle = shuffle)
            #print("then like")
            #print(X_ts_sets)
            model_name = "hierarchical_LSTM_set_" + str(i) + "_run_" + str(j) +\
            '_'.join(str(x) for x in params)
            #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
                   
            write_file_name = "hierarchical_LSTM_" + file_name[:-4] + ".txt"
                    
            hLSTM.write_results(results_path, write_file_name, params, mae, mape, mse)
            
            #K.clear_session()
    
        
       