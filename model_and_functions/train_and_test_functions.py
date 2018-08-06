# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:48:36 2018

@author: iaaraya
"""
import numpy as np

def train_and_test(model, time_steps, lag, epochs, vmin, vmax, X, y, X_ts, y_ts, \
                   batch_size = 1, shuffle = False, verbose = False, overlap = False,\
                   stateful = False):
       
    # Training
    
    if stateful == True:
        
        for i in range(epochs):
            
            model.fit(X, y, batch_size=1, shuffle=False, verbose=0, epochs=1)
            model.reset_states()
            
    else:
        
        model.fit(X, y, batch_size = batch_size, shuffle = shuffle, verbose = False, epochs = epochs)
        
    # Testing 
    
    predicted_vector = np.zeros((24))
    
    X_ts = X_ts.reshape(1, time_steps, lag)
    #print(X_ts)
    #print(y_ts)
    for i in range(24):
                        
        predicted_vector[i] = model.predict(X_ts)
                
        if i != 23:
               
            if overlap == True:
                
                values = np.concatenate((X_ts[0,0,:].flatten(),X_ts[0,1:,-1].flatten()))
                values = np.concatenate((values[1:], predicted_vector[i].flatten()))
                
                X_ts = [values[t:lag + t]  for t in range(time_steps)]
                X_ts = np.array(X_ts).reshape(1, time_steps, lag)
                
            else:
                
                # If used on the LSTM_Ms, time_steps refers to the amount of input values
                # given to the model and lag should always be 1, unless exogenous variables
                # are given as well.
                
                X_ts = np.concatenate((X_ts.flatten()[1:], predicted_vector[i].flatten()))
                
                X_ts = X_ts.reshape(1, time_steps, lag)
    #print(X_ts)
    #print(predicted_vector)                   
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax-vmin) + vmin
    
    mae = np.mean(np.abs(predicted_vector.flatten() - y_ts.flatten()))
    mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    mse = np.mean((predicted_vector.flatten() - y_ts.flatten())**2)
                        
    print(mae)
    return mae, mape, mse, model