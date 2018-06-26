# -*- coding: utf-8 -*-

#from pathlib import Path
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import optimizers, regularizers
import numpy as np
import sys
import os

''' Parameters format "[layer1-layer2-...-layern],lag,time_steps,epochs,l2,learning_rate" '''

def get_params(argv_position):
    
    params = str(sys.argv[argv_position])
    #print(params)
    params = params.split(',')
    
    layers = str(params[0]).strip('[]')
    layers = [int(layer) for layer in layers.split('-')]
    
    lag = int(params[1])
    time_steps = int(params[2])
    epochs = int(params[3])
    l2 = float(params[4])
    learning_rate = float(params[5])
    
    return layers, lag, time_steps, epochs, l2, learning_rate
   
def get_params_gpu(argv_position):
    
    raw_parameters = str(sys.argv[argv_position]).split('--')
    
    parameters_set = []
    
    for params in raw_parameters:

        if params == '':
            
            continue
        
        params = params.split(',')
    
        layers = str(params[0]).strip('[]')
        layers = [int(layer) for layer in layers.split('-')]
        
        lag = int(params[1])
        time_steps = int(params[2])
        epochs = int(params[3])
        l2 = float(params[4])
        learning_rate = float(params[5])
        batch_size = int(params[6])
        verbose = bool(int(params[7]))
        
        parameters_set.append([layers, lag, time_steps, epochs, l2, \
                               learning_rate, batch_size, verbose])
    
    return parameters_set
        

def model(layers, lag, time_steps, l2, learning_rate):
                     
    inputs = Input(batch_shape=(1, time_steps, lag))
    
    dummy_layer = inputs
    
    for i in range(len(layers)):
        
        return_sequences = True
        
        if i == len(layers) - 1:
            
            return_sequences = False
                   
        lstm = LSTM(layers[i], return_sequences=return_sequences, activation='tanh',
                 recurrent_activation='sigmoid', dropout=0.0,
                 recurrent_dropout=0.0, activity_regularizer=regularizers.l2(l2),
                 recurrent_regularizer=regularizers.l2(l2), stateful=True)
    
        dummy_layer = lstm(dummy_layer)
    
    outputs = Dense(1, activation = 'linear')(dummy_layer)

    model = Model(inputs = inputs, outputs = outputs)

    ad = optimizers.Adadelta(lr = learning_rate)
    
    model.compile(optimizer = ad, loss = 'mse')
    
    return model


def model_gpu(layers, lag, time_steps, l2, learning_rate, sets ,runs):
   
    input_layers = []
    output_layers = []
    
    for s in range(sets):
    
        for r in range(runs):
                   
            inputs = Input(batch_shape=(1, time_steps, lag))
            
            dummy_layer = inputs
            
            for i in range(len(layers)):
                
                return_sequences = True
                
                if i == len(layers) - 1:
                    
                    return_sequences = False
                           
                lstm = LSTM(layers[i], return_sequences=return_sequences, activation='tanh',
                         recurrent_activation='sigmoid', dropout=0.0,
                         recurrent_dropout=0.0, activity_regularizer=regularizers.l2(l2),
                         recurrent_regularizer=regularizers.l2(l2), stateful=True)
            
                dummy_layer = lstm(dummy_layer)
            
            outputs = Dense(1, activation = 'linear')(dummy_layer)
        
            input_layers.append(inputs) 
            output_layers.append(outputs)
        
    model = Model(inputs = input_layers, outputs = output_layers)
        
    ad = optimizers.Adadelta(lr = learning_rate)
            
    model.compile(optimizer = ad, loss = 'mse')
    
    return model



""" Only overlaping case is considered """

def train_and_test(model, time_steps, lag, epochs, vmin, vmax, X, y, X_ts, y_ts):
       
    # Training
    
    for i in range(epochs):
        
        model.fit(X, y, batch_size=1, shuffle=False, verbose=0, epochs=1)
        model.reset_states()
        
    # Testing 
    
    predicted_vector = np.zeros((24))
    
    X_ts = X_ts.reshape(1, time_steps, lag)
    
    for i in range(24):
                        
        predicted_vector[i] = model.predict(X_ts)
                
        if i != 23:
               
            values = np.concatenate((X_ts[0,0,:].flatten(),X_ts[0,1:,-1].flatten()))
            values = np.concatenate((values[1:], predicted_vector[i].flatten()))
            
            X_ts = [values[t:lag + t]  for t in range(time_steps)]
            X_ts = np.array(X_ts).reshape(1, time_steps, lag)
                          
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax-vmin) + vmin
    
    mae = np.mean(np.abs(predicted_vector - y_ts))
    mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    mse = np.mean((predicted_vector - y_ts)**2)
                        
    return mae, mape, mse, model

def train_and_test_gpu(model, time_steps, lag, epochs, vmin, vmax,  \
                       X, y, X_ts, y_ts, sets, runs, batch_size, verbose):
       
    # Training
    
    X_sets = []
    y_sets = []
    X_ts_sets = []

    for j in range(sets):
        
        [X_sets.append(X[j]) for i in range(runs)] 
        [y_sets.append(y[j]) for i in range(runs)]
        [X_ts_sets.append(X_ts[j].reshape(1, time_steps, lag)) for i in range(runs)]
    
    
    for i in range(epochs):
        
        model.fit(X_sets, y_sets, batch_size=1, shuffle=False, verbose=verbose, epochs=1)
        model.reset_states()
        
    # Testing 
    
    predicted_vector = np.zeros((24, sets, runs))
    
    for i in range(24):
        
        print(X_ts_sets)
        predicted_vector[i,:,:] = model.predict(X_ts_sets)
        print(predicted_vector[i,:,:])
                
        if i != 23:
            
            for s in range(sets):
                
                for k in range(runs):
                        
                    temporal_X = X_ts_sets[s*runs + k]
                    
                    values = np.concatenate((temporal_X[0,0,:].flatten(), \
                                             temporal_X[0,1:,-1].flatten()))
                    values = np.concatenate((values[1:], predicted_vector[i,s*runs + k].flatten()))
            
                    temporal_X = [values[t:lag + t]  for t in range(time_steps)]
                    temporal_X = np.array(temporal_X).reshape(1, time_steps, lag)
                                       
                    X_ts_sets[s*runs + k] = temporal_X
            
    
    mae = np.zeros((24,sets,runs))
    mape = np.zeros((24,sets,runs))
    mse = np.zeros((24,sets,runs))
    
    for s in range(sets):
        
        y_ts[s] = y_ts[s] * (vmax[s] - vmin[s]) + vmin[s]
        
        for r in range(runs):
                        
            predicted = predicted_vector[:, s*runs + r] * (vmax[s] - vmin[s]) + vmin[s] 
            mae[:,s,r] = np.abs(predicted.flatten() - y_ts[s].flatten())
            mape[:,s,r] = np.abs((predicted.flatten() - y_ts[s].flatten() )/y_ts[s].flatten())*100
            mse[:,s,r] = (predicted.flatten() - y_ts[s].flatten())**2
    
    
    mae = np.mean(mae,axis=0)
    mape = np.mean(mape,axis=0)
    mse = np.mean(mse, axis=0)
                      
    print(np.mean(mae))
    return mae, mape, mse, model

""" Results are written as "params mean_mae mean_mape mean_mse std_mae std_mape std_mse" """

def write_results(path,name,params,mae,mape,mse):
    
    for i in range(10):
            
            my_file = path+str(i)+name
            
            if not os.path.exists(my_file):
            #if not my_file.is_file():
                
                f = open(path + str(i) + name, "a")
                f.write("layers lag time_steps epochs l2 learning_rate mean_mae \
                        mean_mape mean_mse std mae std_mape std_mse \n")
            
            else:
                
                f = open(path + str(i) + name, "a")
                    
            mean_mae, std_mae = str(np.mean(mae[i,:])), str(np.std(mae[i,:]))
            mean_mape, std_mape = str(np.mean(mape[i,:])), str(np.std(mape[i,:]))
            mean_mse, std_mse = str(np.mean(mse[i,:])), str(np.std(mse[i,:]))

            f.write('{} {} {} {} {} {} {} \n'.format(', '.join(str(x) for x in params) \
                    , mean_mae, mean_mape, mean_mse, std_mae, std_mape, std_mse) )
            
            f.close()
                      
