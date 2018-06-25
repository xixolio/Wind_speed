# -*- coding: utf-8 -*-
import sys
#sys.path.append('../')
sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/data/')
sys.path.append('C:/Users/iaaraya/Documents/CIARP/Wind_speed/model_and_functions/')

#from data_processing.data_processing import get_data
from data_processing import get_data

#import model_and_functions.simple_LSTM as sLSTM
import hierarchical_LSTM as hLSTM
import numpy as np

if __name__ == "__main__":
    
    path = sys.argv[1]
    
    file_name = sys.argv[2]
    
    print("Printing path and file_name")
    print("path: " + path)
    print("file_name: " + file_name)
    
    runs = 5
    
    sets = 10
       
    lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
    epochs, l2 = hLSTM.get_params(3)

    params = [lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
               epochs, l2]
    
    print("Printing parameters")
    print("lstm_nodes: " + str(lstm_nodes))
    print("dense_nodes: " + str(dense_nodes))
    print("lags: " + str(lags))
    print("time_steps: " + str(time_steps))
    print("epochs: " + str(epochs))
    print("processed_scales: " + str(processed_scales))
    print("l2: " + str(l2))
    
    training_inputs_sets = []
    testing_inputs_sets = []
    
    for i in range(len(lags)):
        
        training_inputs, testing_inputs, training_outputs, testing_outputs,\
        vmins, vmaxs = get_data(path, file_name, time_steps[i], lags[i])
        
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
        
        # Changing training data length to the minimum
        
        print("Printing data shapes")
    
        for j in range(len(lags)):
            
             X.append(training_inputs_sets[j][i][-min_data_len:])
             X_ts.append(testing_inputs_sets[j][i])
             
             print("X shape: " + str(j) + " " + str(X[j].shape))
             print("X_ts shape: " + str(j) + " " + str(X_ts[j].shape))
             
             X[j] = X[j][-100:]
             X_ts[j] = X_ts[j][-100:]

        y = training_outputs[i][-min_data_len:]
        y_ts = testing_outputs[i]
        
        print("y shape: " + str(y.shape))
        print("y_ts shape: " + str(y_ts.shape))
        print("vmin: " + str(vmins[i]))
        print("vmax: " + str(vmaxs[i]))
        
        y = y[-100:]
        y_ts = y_ts[-100:]
        
        for j in range(runs):
            
            model = hLSTM.model(lags, time_steps, processed_scales, \
                                dense_nodes, lstm_nodes, l2)
            
            mae[i,j], mape[i,j], mse[i,j], model = hLSTM.train_and_test(model, time_steps, lags, \
                                                  epochs, vmins[i], vmaxs[i],     \
                                                  X, y, X_ts.copy(), y_ts)
            
            model_name = "hierarchical_LSTM_set_" + str(i) + "_run_" + str(j) +\
            '_'.join(str(x) for x in params)
            #model.save("/user/i/iaraya/CIARP/Wind_speed/models/" + model_name + ".h5")
           
    path = "../results/"
    write_file_name = "hierarchical_LSTM_test_" + file_name[:-4] + ".txt"
            
    hLSTM.write_results(path, write_file_name, params, mae, mape, mse)
        