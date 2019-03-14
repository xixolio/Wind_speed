# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:07:18 2018

@author: iaaraya
"""
import os
#os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
#sys.path.append('/user/i/iaraya/Wind_speed/data/')
#sys.path.append('/user/i/iaraya/Wind_speed/model_and_functions/')
#sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/data/')
#sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/model_and_functions/')

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='1' # gpu='0' o gpu='1'
###################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True#Utiliza la memoria que necesita de manera dinamica, puede ser o no en bloque.
config.gpu_options.per_process_gpu_memory_fraction = 0.2#20%de la ram,
session = tf.Session(config=config)
###################################



import copy
import numpy as np
np.random.seed(46)

if __name__ == "__main__":
    
    model = sys.argv[1]
        
    path = sys.argv[2]
        
    file_name = sys.argv[3]
    
    experiment = sys.argv[5]
    
    if experiment == 'test':
        set_index = int(sys.argv[6])
    
    data_path = path+"data/"
    
    functions_path = path+"model_and_functions/"
    results_path = path+"results/"
    sys.path.append(data_path)
    sys.path.append(functions_path)
    
    from data_processing import get_data
    import simple_LSTM as sLSTM
    import LSTM_Ms as Ms
    import write_results as wr
    import get_params as gp
    import train_and_test_functions as trf
    import persistence
        
    if model == "simple_LSTM":
      
        runs = 1
        sets = 5
        if experiment == 'test':
            runs = 5
        layers, lag, time_steps, epochs, l2, learning_rate, batch_size = gp.get_params(4)
    
        params = [layers, lag, time_steps, epochs, l2, learning_rate, batch_size]
        
        training_inputs, validation_inputs,testing_inputs, training_outputs, validation_outputs,\
        testing_outputs,vmins, vmaxs = get_data(data_path, file_name, time_steps, lag)
            
        mae = np.zeros((sets, runs))
        #mape = np.zeros((sets, runs))
        mse = np.zeros((sets, runs))
        h_mae = np.zeros((sets,runs,24))
        h_mse = np.zeros((sets,runs,24))
        
        if experiment == 'test':
            sets = 1
        
        for i in range(sets):
            
            if experiment == 'test':
                i = set_index
                
            X = training_inputs[i]
            X_val = validation_inputs[i]
            X_ts = testing_inputs[i]
            y = training_outputs[i]
            y_val = validation_outputs[i]
            y_ts = testing_outputs[i]
            
            for j in range(runs):
                
                mod = sLSTM.model(layers, lag, time_steps, l2, learning_rate)
                
                if experiment == 'validation':
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, time_steps, lag, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_val), copy.deepcopy(y_val),  batch_size = batch_size, \
                                                          shuffle = True, overlap = True, experiment = experiment)
                    write_file_name = str(model) + '_' + file_name[:-4] + "set_"+str(i)+".txt"
                elif experiment == 'test':
                    X = np.concatenate((X,X_val),axis=0)
                    y = np.concatenate((y,y_val[:,0]),axis=0)
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, time_steps, lag, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_ts), copy.deepcopy(y_ts),  batch_size = batch_size, \
                                                          shuffle = True, overlap = True, experiment = experiment)
                    write_file_name = str(model) + '_test_' + file_name[:-4] + "set_"+str(i)+".txt"
            wr.write_result(results_path, write_file_name, params, mae[i], mse[i],h_mae[i],h_mse[i],epoch)
        
    elif model == 'LSTM_Ms' or model == 'LSTM_Ms_pool' or model == 'LSTM_Ms_locally' \
    or model == 'LSTM_Ms_return' or model == 'SRNN_Ms_return':
        
        runs = 1
        sets = 5
        if experiment == 'test':
            runs = 5
        
        lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
        epochs, l2, batch_size, shuffle, final_nodes = gp.get_params_Ms(4)
    
        params = [lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                   epochs, l2, batch_size, shuffle, final_nodes]
        
        max_input_values = np.max([lags[i]*time_steps[i] for i in range(len(lags))])
        
            
        training_inputs, validation_inputs, testing_inputs, training_outputs, validation_outputs,\
        testing_outputs,vmins, vmaxs = get_data(data_path, file_name, max_input_values, 1, overlap=False)
        
        mae = np.zeros((sets, runs))
        h_mae = np.zeros((sets,runs,24))
        #mape = np.zeros((sets, runs))
        mse = np.zeros((sets, runs))
        h_mse = np.zeros((sets,runs,24))
        
        if experiment == 'test':
            sets = 1
        
        for i in range(sets):
            
            if experiment == 'test':
                i = set_index
                
            X = training_inputs[i]
            X_val = validation_inputs[i]
            X_ts = testing_inputs[i]
            y = training_outputs[i]
            y_val = validation_outputs[i]
            y_ts = testing_outputs[i]
            
            for j in range(runs):
                
                if model == 'LSTM_Ms':
                    
                    mod = Ms.LSTM_Ms(lags, time_steps, processed_scales, \
                                        dense_nodes, lstm_nodes, l2,final_nodes)
                    
                elif model == 'LSTM_Ms_pool':
                    
                    mod = Ms.LSTM_Ms_pool(lags, time_steps, processed_scales, \
                                        dense_nodes, lstm_nodes, l2,final_nodes)
                    
                elif model == 'LSTM_Ms_locally':
                   
                    mod = Ms.LSTM_Ms_locally(lags, time_steps, processed_scales, \
                                        dense_nodes, lstm_nodes, l2)
                    
                elif model == 'LSTM_Ms_return':
                    
                    mod = Ms.LSTM_Ms_return(lags, time_steps, processed_scales, \
                                        dense_nodes, lstm_nodes, l2, final_nodes)
                    
                elif model == 'SRNN_Ms_return':
                    
                    mod = Ms.SRNN_Ms_return(lags, time_steps, processed_scales, \
                                        dense_nodes, lstm_nodes, l2, final_nodes)
                    
                if experiment == 'validation': 
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, max_input_values, 1, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_val), copy.deepcopy(y_val),  batch_size = batch_size, \
                                                          shuffle = shuffle,experiment = experiment)
                    write_file_name = str(model) + '_' + file_name[:-4] + "set_"+str(i)+".txt"
                    
                elif experiment == 'test':
                    X = np.concatenate((X,X_val),axis=0)
                    y = np.concatenate((y,y_val[:,0]),axis=0)
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, max_input_values, 1, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_ts), copy.deepcopy(y_ts),  batch_size = batch_size, \
                                                          shuffle = shuffle,experiment = experiment)
                    write_file_name = str(model) + '_test_' + file_name[:-4] + "set_"+str(i)+".txt"
            
            wr.write_result(results_path, write_file_name, params, mae[i], mse[i],h_mae[i],h_mse[i],epoch)
            
        
    elif model == 'Conv' or model == 'TDNN' or model == 'TDNN_l':
        
        runs = 1
        sets = 5
        if experiment == 'test':
            runs = 5
       
           
        lags, dense_nodes, input_length, final_nodes, epochs, l2,\
            batch_size, shuffle = gp.get_params_Conv(4)
    
        params = [lags, dense_nodes, input_length, final_nodes, epochs, l2,\
            batch_size, shuffle]
        
        #max_input_values = np.max([lags[i]*time_steps[i] for i in range(len(lags))])
        
        training_inputs, validation_inputs, testing_inputs, training_outputs, validation_outputs,\
        testing_outputs,vmins, vmaxs = get_data(data_path, file_name, input_length, 1, overlap=False)
        
            
        #training_inputs, testing_inputs, training_outputs, testing_outputs,\
        #vmins, vmaxs = get_data(path, file_name, input_length, 1, overlap=False)
        
        mae = np.zeros((sets, runs))
        #mape = np.zeros((sets, runs))
        mse = np.zeros((sets, runs))
        
        h_mae = np.zeros((sets,runs,24))
        h_mse = np.zeros((sets,runs,24))
        
        if experiment == 'test':
            sets = 1
        
        for i in range(sets):
            
            if experiment == 'test':
                i = set_index
            
            X = training_inputs[i]
            X_val = validation_inputs[i]
            X_ts = testing_inputs[i]
            y = training_outputs[i]
            y_val = validation_outputs[i]
            y_ts = testing_outputs[i]
            
            for j in range(runs):
                
                if model == 'Conv':
                    
                    mod = Ms.Conv(lags, dense_nodes, input_length, l2, final_nodes)
                
                elif model == 'TDNN':
                    
                    mod = Ms.TDNN(lags, dense_nodes, input_length, l2, final_nodes)

                elif model == 'TDNN_l':
                    
                    mod = Ms.TDNN_locally(lags, dense_nodes, input_length, l2, final_nodes)
                
                if experiment == 'validation':
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, input_length, 1, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_val), copy.deepcopy(y_val),  batch_size = batch_size, \
                                                          shuffle = shuffle, experiment = experiment)
                    write_file_name = str(model) + '_' + file_name[:-4] + "set_"+str(i)+".txt"
                    
                elif experiment == 'test':
                    X = np.concatenate((X,X_val),axis=0)
                    y = np.concatenate((y,y_val[:,0]),axis=0)
                    mae[i,j], mse[i,j],h_mae[i,j,:],h_mse[i,j,:], epoch = trf.train(mod, input_length, 1, \
                                                          epochs, vmins[i], vmaxs[i],     \
                                                          X, y, copy.deepcopy(X_ts), copy.deepcopy(y_ts),  batch_size = batch_size, \
                                                          shuffle = shuffle,experiment = experiment)
                    write_file_name = str(model) + '_test_' + file_name[:-4] + "set_"+str(i)+".txt"
            
            wr.write_result(results_path, write_file_name, params, mae[i], mse[i],h_mae[i],h_mse[i],epoch)
            
        
    elif model == "persistence":
        
        sets = 5
        training_inputs, validation_inputs, testing_inputs, training_outputs, validation_outputs,\
        testing_outputs,vmins, vmaxs = get_data(data_path, file_name, 24, 1, overlap=False)
        
        mae = np.zeros((sets,1))
        mse = np.zeros((sets,1))
        
        h_mae = np.zeros((sets,1,24))
        h_mse = np.zeros((sets,1,24))
        
        for i in range(sets):
            
            X_ts = testing_inputs[i]
            y_ts = testing_outputs[i]

            mae[i], mse[i],h_mae[i,:],h_mse[i,:] = persistence.train_and_test(vmins[i], vmaxs[i], X_ts, y_ts)
            
            print(h_mae[i])
            write_file_name = 'persistence_test_' + file_name[:-4] + "set_"+str(i)+".txt"
            wr.write_result(results_path, write_file_name, [24], mae[i], mse[i],h_mae[i],h_mse[i],0)
            
        
        
       