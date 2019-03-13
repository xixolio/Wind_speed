# -*- coding: utf-8 -*-

import numpy as np
import os

def train_and_test(vmin, vmax, X_ts, y_ts):
    
    predicted_vector = X_ts.reshape(-1,24)
    
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax - vmin) + vmin
    
    h_mae = np.mean(np.abs(predicted_vector - y_ts),axis=0)
    mae = np.mean(h_mae)
    #mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    h_mse = np.mean((predicted_vector.flatten() - y_ts.flatten())**2,axis=0)
    mse = np.mean(h_mse)
                        
    return mae, mse,h_mae,h_mse
    

