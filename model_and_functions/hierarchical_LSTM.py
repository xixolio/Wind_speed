# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras import regularizers

import keras

import numpy as np
import sys
import os

''' Parameters format "[lag1-lag2-...-lagn],[time_step1-time_step2-...-time_stepn],
    [dense_nodes1-...-dense_nodesn],[lstm_nodes1-...-lstm_nodesn],[index1-...-indexm]
    ,epochs,l2" '''

def get_params(argv_position):
    
    params = str(sys.argv[argv_position]).split(',')

    lags = str(params[0]).strip('[]')
    lags = [int(lag) for lag in lags.split('-')]

    time_steps = str(params[1]).strip('[]')
    time_steps = [int(ts) for ts in time_steps.split('-')]
    
    dense_nodes = str(params[2]).strip('[]')
    dense_nodes = [int(dn) for dn in dense_nodes.split('-')]
    
    lstm_nodes = str(params[3]).strip('[]')
    lstm_nodes= [int(ln) for ln in lstm_nodes.split('-')]

    processed_scales = str(params[4]).strip('[]')
    processed_scales = [int(index) for index in processed_scales.split('-')]

    epochs = int(params[5])
    
    l2 = float(params[6])
    
    return lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
            epochs, l2
            
def get_params_gpu(argv_position):
    
    raw_parameters = str(sys.argv[argv_position]).split('--')
    
    parameters_set = []
    
    for params in raw_parameters:

        if params == '':
            
            continue
        
        params = params.split(',')
        
        lags = str(params[0]).strip('[]')
        lags = [int(lag) for lag in lags.split('-')]
    
        time_steps = str(params[1]).strip('[]')
        time_steps = [int(ts) for ts in time_steps.split('-')]
        
        dense_nodes = str(params[2]).strip('[]')
        dense_nodes = [int(dn) for dn in dense_nodes.split('-')]
        
        lstm_nodes = str(params[3]).strip('[]')
        lstm_nodes= [int(ln) for ln in lstm_nodes.split('-')]
    
        processed_scales = str(params[4]).strip('[]')
        processed_scales = [int(index) for index in processed_scales.split('-')]
    
        epochs = int(params[5])
        
        l2 = float(params[6])
        
        batch_size = int(params[7])
        
        shuffle = bool(int(params[8]))
        
        verbose = bool(int(params[9]))
        
        parameters_set.append([lags, time_steps, dense_nodes, lstm_nodes, \
                               processed_scales, epochs, l2, batch_size, shuffle,\
                               verbose])
    
    return parameters_set


def model(lags, time_steps, processed_scales, dense_nodes, lstm_nodes, l2):
    
    input_layers = []
    lstm_layers = []
    dense_layers = []
    
    # Dense layers instantiated and saved
    
    for i in range(len(lags)):
            
        if lags[i] > 1:
               
            dense_layers.append(Dense(dense_nodes[i],activation='sigmoid'))
    
    
    for i in range(len(lags)):
        
        time_step = time_steps[i]
        lag = lags[i]
        temporal_input = Input(shape=(time_step,lag))

        input_layers.append(temporal_input)
    
        # Scales and LSTMs which proccess them are built
        
        if lag > 1:
            
            dummy_layer = temporal_input
            
            # i represents the current scale being worked on. This loops takes
            # all previous dense layers and puts them together to build that
            # scale.
            
            for j in range(i): 
                
                # How many time steps in the current scale have to be taken 
                # into account to consequently build the current LSTM. Example:
                # scale 3 is being processed for 3 time steps and is made of 6 values
                # each, so 18 values total. Previous scale is made of 2 values
                # so its needed that 3 * 6 / 2 time steps, 9, be computed at
                # that scale in order to give 18 values to the next one.
                
                
                current_time_steps = int(time_step * lag/ lags[j+1])
                
                # The dimension of the values processed at each time step. Example:
                # same as before, but say we are building the LSTM for the last scale.
                # If  scale is made of 6 values and the previous one by 2,
                # but those 2 values were processed by a dense layer with 5 nodes,
                # then, each of the 18 time steps are composed of 6/2 = 3 values
                # multiplied by the 5 nodes each of the 2 were transformed to.
                
                current_dimension = int(dense_nodes[j]*lags[j+1]/lags[j])
    
                reshaped_layer = Reshape((current_time_steps, current_dimension))(dummy_layer)
                dummy_layer = TimeDistributed(dense_layers[j])(reshaped_layer)
                
            # Finally, after all the previous dense structure was created
            # the LSTM is built using the final dummy_layer.
            
            lstm = LSTM(lstm_nodes[i],activation='sigmoid',
                    recurrent_activation='sigmoid',
                    activity_regularizer=regularizers.l2(l2),
                    recurrent_regularizer=regularizers.l2(l2))(dummy_layer)
            
        else:
            
            lstm = LSTM(lstm_nodes[i],activation='sigmoid',
                recurrent_activation='sigmoid',
                activity_regularizer=regularizers.l2(l2),
                recurrent_regularizer=regularizers.l2(l2))(temporal_input)
            
        lstm_layers.append(lstm)
    
    if len(lstm_layers) > 1:
        
        layers_to_concatenate = [lstm_layers[index] for index in processed_scales]
        concatenated = keras.layers.concatenate(layers_to_concatenate)
        #concatenated = lstms[-1]
        #concatenated = Flatten()(dummy)
        #concatenated = Dense(5,activation='sigmoid')(concatenated)
    
    else: 
        
        concatenated = lstm_layers[0]
        
    outputs = Dense(1)(concatenated)
    model = Model(inputs=input_layers,outputs=outputs)
    model.compile(loss='mse',optimizer='adam')
    
    return model

def model_gpu(lags, time_steps, processed_scales, dense_nodes, lstm_nodes, l2, runs):
    
    model_inputs = []
    model_outputs = []
    
    for k in range(runs):
        
        input_layers = []
        lstm_layers = []
        dense_layers = []
        
        # Dense layers instantiated and saved
        
        for i in range(len(lags)):
                
            if lags[i] > 1:
                   
                dense_layers.append(Dense(dense_nodes[i],activation='sigmoid'))
        
        
        for i in range(len(lags)):
            
            time_step = time_steps[i]
            lag = lags[i]
            temporal_input = Input(shape = (time_step, lag))
    
            input_layers.append(temporal_input)
        
            # Scales and LSTMs which proccess them are built
            
            if lag > 1:
                
                dummy_layer = temporal_input
                
                # i represents the current scale being worked on. This loops takes
                # all previous dense layers and puts them together to build that
                # scale.
                
                for j in range(i): 
                    
                    # How many time steps in the current scale have to be taken 
                    # into account to consequently build the current LSTM. Example:
                    # scale 3 is being processed for 3 time steps and is made of 6 values
                    # each, so 18 values total. Previous scale is made of 2 values
                    # so its needed that 3 * 6 / 2 time steps, 9, be computed at
                    # that scale in order to give 18 values to the next one.
                    
                    
                    current_time_steps = int(time_step * lag/ lags[j+1])
                    
                    # The dimension of the values processed at each time step. Example:
                    # same as before, but say we are building the LSTM for the last scale.
                    # If  scale is made of 6 values and the previous one by 2,
                    # but those 2 values were processed by a dense layer with 5 nodes,
                    # then, each of the 18 time steps are composed of 6/2 = 3 values
                    # multiplied by the 5 nodes each of the 2 were transformed to.
                    
                    current_dimension = int(dense_nodes[j]*lags[j+1]/lags[j])
        
                    reshaped_layer = Reshape((current_time_steps, current_dimension))(dummy_layer)
                    dummy_layer = TimeDistributed(dense_layers[j])(reshaped_layer)
                    
                # Finally, after all the previous dense structure was created
                # the LSTM is built using the final dummy_layer.
                
                lstm = LSTM(lstm_nodes[i],activation='sigmoid',
                        recurrent_activation='sigmoid',
                        activity_regularizer=regularizers.l2(l2),
                        recurrent_regularizer=regularizers.l2(l2))(dummy_layer)
                
            else:
                
                lstm = LSTM(lstm_nodes[i],activation='sigmoid',
                    recurrent_activation='sigmoid',
                    activity_regularizer=regularizers.l2(l2),
                    recurrent_regularizer=regularizers.l2(l2))(temporal_input)
                
            lstm_layers.append(lstm)
        
        if len(lstm_layers) > 1:
            
            layers_to_concatenate = [lstm_layers[index] for index in processed_scales]
            concatenated = keras.layers.concatenate(layers_to_concatenate)
            #concatenated = lstms[-1]
            #concatenated = Flatten()(dummy)
            #concatenated = Dense(5,activation='sigmoid')(concatenated)
        
        else: 
            
            concatenated = lstm_layers[0]
            
        outputs = Dense(1)(concatenated)
        
        model_outputs.append(outputs)
        #model = Model(inputs=input_layers,outputs=outputs)
        
    
    model = Model(inputs=model_inputs,outputs=model_outputs)
    model.compile(loss='mse',optimizer='adam')
    

    return model


def train_and_test(model, time_steps, lags, epochs, vmin, vmax, X, y, X_ts, y_ts, \
                   batch_size = 1, shuffle = False, verbose = False ):
       
    # Training
    
    for i in range(epochs):
        
        model.fit(X, y, batch_size=batch_size, shuffle=shuffle, verbose = verbose,\
                  epochs=1)
        
        model.reset_states()
        
    # Testing 
    
    predicted_vector = np.zeros((24))
        
    for i in range(24):
                        
        predicted_vector[i] = model.predict(X_ts)
                
        if i != 23:
            
            for j in range(len(lags)):
                
                X_ts[j] = np.concatenate((X_ts[j].flatten()[1:], \
                    predicted_vector[i].flatten()))
                
                X_ts[j] = X_ts[j].reshape(1, time_steps[j], lags[j])
                          
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax - vmin) + vmin
    
    mae = np.mean(np.abs(predicted_vector - y_ts))
    mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    mse = np.mean((predicted_vector - y_ts)**2)
                        
    return mae, mape, mse, model

def train_and_test_gpu(model, time_steps, lags, epochs, vmin, vmax, X, y, X_ts, y_ts,runs  \
                   batch_size = 1, shuffle = False, verbose = False):
       
    
    # Output is replicated for gpu trick      
        
    # Training
    
    for i in range(epochs):
        
        model.fit(X, [y for j in range(runs)], batch_size=batch_size,\
                      shuffle=shuffle, verbose = verbose, epochs=1)
        
        model.reset_states()
        
    # Testing 
    
    predicted_vector = np.zeros((24,runs))
        
    for i in range(24):
                        
        predicted_vector[i,:] = model.predict(X_ts)
                
        if i != 23:
            
            for j in range(len(lags)):
                
                X_ts[j] = np.concatenate((X_ts[j].flatten()[1:], \
                    predicted_vector[i].flatten()))
                
                X_ts[j] = X_ts[j].reshape(1, time_steps[j], lags[j])
                          
    predicted_vector = predicted_vector * (vmax - vmin) + vmin 
    y_ts = y_ts * (vmax - vmin) + vmin
    
    mae = np.mean(np.abs(predicted_vector - y_ts))
    mape = np.mean(np.abs((predicted_vector - y_ts )/y_ts)*100)
    mse = np.mean((predicted_vector - y_ts)**2)
                        
    return mae, mape, mse, model


""" Results are written as "params mean_mae mean_mape mean_mse std_mae std_mape std_mse" """

def write_results(path,name,params,mae,mape,mse):
    
    for i in range(10):
            
            my_file = path+str(i)+name
            
            if not os.path.exists(my_file):
            #if not my_file.is_file():
                
                f = open(path + str(i) + name, "a")
                f.write("lags time_steps dense_nodes lstm_nodes processed scales\
                        epochs l2 learning_rate mean_mae mean_mape mean_mse \
                        std mae std_mape std_mse \n")
            
            else:
                
                f = open(path + str(i) + name, "a")
                    
            mean_mae, std_mae = str(np.mean(mae[i,:])), str(np.std(mae[i,:]))
            mean_mape, std_mape = str(np.mean(mape[i,:])), str(np.std(mape[i,:]))
            mean_mse, std_mse = str(np.mean(mse[i,:])), str(np.std(mse[i,:]))

            f.write('{} {} {} {} {} {} {} \n'.format(', '.join(str(x) for x in params) \
                    , mean_mae, mean_mape, mean_mse, std_mae, std_mape, std_mse) )
            
            f.close()