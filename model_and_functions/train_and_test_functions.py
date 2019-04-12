# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:48:36 2018

@author: iaaraya
"""
import numpy as np
import copy

def train(model, time_steps, lag, epochs, vmin, vmax, X, y, X_ts, y_ts, \
                   batch_size = 32, shuffle = True, verbose = False, overlap = False,\
                   stateful = False, experiment = 'validation'):
    
    best_mae = None
    count = 0
    tolerance = 5
    #best_results = []
    if experiment == 'validation':
        for i in range(epochs):
                model.fit(X, y, batch_size = batch_size, shuffle = shuffle, verbose = False, epochs = 1)
                mae,mse,h_mae,h_mse,_ = test(model,time_steps,lag,epochs,vmin,vmax,copy.deepcopy(X_ts),copy.deepcopy(y_ts),overlap=overlap)
                #print(mae)
                if best_mae:
                    if mae < best_mae:
                        count = 0
                        best_mae = mae
                        best_results = [mae,mse,h_mae,h_mse,i]
                    else:
                        count += 1
                    if count == tolerance:
                        break
                else:
                    best_mae = mae
                    best_results = [mae,mse,h_mae,h_mse,i]
        [mae,mse,h_mae,h_mse,epoch] = best_results
    elif experiment == 'test':
        model.fit(X, y, batch_size = batch_size, shuffle = shuffle, verbose = False, epochs = epochs)
        mae,mse,h_mae,h_mse,_ = test(model,time_steps,lag,epochs,vmin,vmax,copy.deepcopy(X_ts),copy.deepcopy(y_ts),overlap=overlap)
        print(mae)
        epoch = epochs
    return mae,mse,h_mae,h_mse,epoch
            
    
def test(model, time_steps, lag, epochs, vmin, vmax, X_ts, y_ts, \
                   batch_size = 1, shuffle = False, verbose = False, overlap = False,\
                   stateful = False):
    N = len(X_ts)
    predicted_vector = np.zeros((N,24))
    
    X_ts = X_ts.reshape(-1, time_steps, lag)
    #print(overlap)
    for i in range(24):
        
        predicted_vector[:,i] = model.predict(X_ts).flatten()
        #print(X_ts[0,-1,:])
        #print(predicted_vector[0,i])
        if i != 23:
               
            if overlap == True:
                
                X_ts[:,:,:-1] = X_ts[:,:,1:]
                X_ts[:,:-1,-1] = X_ts[:,1:,-1]
                X_ts[:,-1,-1] = predicted_vector[:,i]
                
            else:
                
                X_ts[:,:-1,0] = X_ts[:,1:,0]
                X_ts[:,-1,0] = predicted_vector[:,i]
                   
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax-vmin) + vmin
    
    #print(predicted_vector.shape)
    #print(y_ts.shape)
    h_mae = np.mean(np.abs(predicted_vector - y_ts),axis=0)
    mae = np.mean(h_mae)
    #mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    h_mse = np.mean((predicted_vector.flatten() - y_ts.flatten())**2,axis=0)
    mse = np.mean(h_mse)
            
    #print(h_mae)            
    #print(mae)
    
    return mae, mse,h_mae,h_mse,model
    
    
def train_and_test(model, time_steps, lag, epochs, vmin, vmax, X, y, X_ts, y_ts, \
                   batch_size = 1, shuffle = False, verbose = False, overlap = False,\
                   stateful = False):
       
    # Training
    
    if stateful == True:
        
        for i in range(epochs):
            
            model.fit(X, y, batch_size=1, shuffle=False, verbose=0, epochs=1)
            model.reset_states()
            
    else:
        
        for i in range(epochs):
            model.fit(X, y, batch_size = batch_size, shuffle = shuffle, verbose = False, epochs = 1)
        
    # Testing 
    
    N = len(X_ts)
    predicted_vector = np.zeros((N,24))
    
    X_ts = X_ts.reshape(-1, time_steps, lag)
    #print(X_ts)
    #print(y_ts)
    #print(X_ts[0])
    for i in range(24):
                        
        predicted_vector[:,i] = model.predict(X_ts).flatten()
                
        if i != 23:
               
            if overlap == True:
                
                #values = np.concatenate((X_ts[0,0,:].flatten(),X_ts[0,1:,-1].flatten()))
                #values = np.concatenate((values[1:], predicted_vector[i].flatten()))
                
                X_ts[:,:,:-1] = X_ts[:,:,1:]
                X_ts[:,:-1,-1] = X_ts[:,1:,-1]
                X_ts[:,-1,-1] = predicted_vector[:,i]
                #print(X_ts[0])
                #values = X_ts[:,:,1:]
                
                #X_ts = [values[t:lag + t]  for t in range(time_steps)]
                #X_ts = np.array(X_ts).reshape(1, time_steps, lag)
                
            else:
                
                # If used on the LSTM_Ms, time_steps refers to the amount of input values
                # given to the model and lag should always be 1, unless exogenous variables
                # are given as well.
                
                #X_ts = np.concatenate((X_ts.flatten()[1:], predicted_vector[i].flatten()))
                
                X_ts[:,:-1,0] = X_ts[:,1:,0]
                X_ts[:,-1,0] = predicted_vector[:,i]
                #print(predicted_vector[1,i])
                #print(X_ts[1])
                #X_ts = X_ts.reshape(1, time_steps, lag)
    #print(X_ts)
    #print(predicted_vector)                   
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax-vmin) + vmin
    
    #print(predicted_vector.shape)
    #print(y_ts.shape)
    h_mae = np.mean(np.abs(predicted_vector - y_ts),axis=0)
    mae = np.mean(h_mae)
    #mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    h_mse = np.mean((predicted_vector.flatten() - y_ts.flatten())**2,axis=0)
    mse = np.mean(h_mse)
            
    #print(h_mae)            
    #print(mae)
    
    return mae, mse,h_mae,h_mse,model