# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:07:18 2018

@author: iaaraya
"""

from data_processing/data_processing import get_data
import model_and_functions/simple_LSTM as sLSTM

import sys

if __name__ == "__main__":
    
    
    model = sys.argv[1]
    
    path = sys.argv[2]
    
    file_name = sys.argv[3]
    
    runs = 5
    
    sets = 10
    
    if model == "simple_LSTM":
        
        layers, lag, time_steps, epochs, l2, learning_rate = sLSTM.get_params()
        
        X_tr, X_ts, y_tr, y_ts, vmins, vmax = get_data(path, file_name, time_steps, lag)

        for i in range(sets):
            
            for j in range(runs):
                
                model = sLSTM.model(layers, lag, time_steps, l2, learning_rate)
        
                mae, mape, mse, model = train_and_test(model, time_steps, lag, \
                                                      epochs, vmin, vmax,     \
                                                      X, y, X_ts, y_ts)
                
                
        
        