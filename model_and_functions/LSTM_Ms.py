# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:56:29 2018

@author: iaaraya
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, \
Reshape, Lambda, Conv1D, MaxPooling1D, Flatten
from keras import regularizers, optimizers

import keras

import numpy as np
import sys
import os

def z_n(x, position):
    
    return x[:, -position:, :]

#lambda_layer = Lambda(return_specific, arguments = {'position': 1})(dense)
#dense = Dense(1)(lambda_layer)

def LSTM_Ms(lags, time_steps, processed_scales, dense_nodes, lstm_nodes, l2):

    number_layers = len(lags)
    
    # we get the max number of values required by the model
    
    max_input_values = np.max([lags[i]*time_steps[i] for i in range(number_layers)])
    
    inputs = Input(shape = (max_input_values, 1))
    
    dense_layers = []
    
    dense_layers.append(inputs)
    
    for i in range(1, number_layers):
        
        strides =  lags[i]//lags[i-1]
        
        conv = Conv1D(filters = dense_nodes[i], kernel_size = strides, strides = strides, \
               activation = 'sigmoid', use_bias = True)(dense_layers[i-1])
        
        dense_layers.append(conv)
        
    lstm_layers = []
    
    for scale in processed_scales:
        
        lambda_layer = Lambda(z_n, arguments = {'position': time_steps[scale]})(dense_layers[scale])
        
        lstm = LSTM(lstm_nodes[scale], activation='sigmoid', recurrent_activation='sigmoid',\
                activity_regularizer=regularizers.l2(l2), \
                recurrent_regularizer=regularizers.l2(l2))(lambda_layer)
        
        lstm_layers.append(lstm)
        
    if len(lstm_layers) > 1:
        
        concatenated = keras.layers.concatenate(lstm_layers)
        
    else:
        
        concatenated = lstm_layers[0]
        
    outputs = Dense(1)(concatenated)
    #outputs = dense_layers[2]
    #outputs2 = dense_layers[1]
    #outputs = concatenated
    model = Model(inputs = inputs, outputs = outputs)
    #model2 = Model(inputs = inputs, outputs = outputs2)
    ad = optimizers.Adadelta(lr = 0.05)
    
    model.compile(loss = 'mse', optimizer = ad)
    #model2.compile(loss = 'mse', optimizer = ad)
    #model.compile(loss = 'mse', optimizer = "sgd")
    
    return model

def Conv(lags, dense_nodes, input_length, l2, final_nodes):

    number_layers = len(lags)
    
    # we get the max number of values required by the model
    
    #max_input_values = np.max([lags[i]*time_steps[i] for i in range(number_layers)])
    
    inputs = Input(shape = (input_length, 1))
    
    dense_layers = []
    
    dense_layers.append(inputs)
    
    for i in range(1, number_layers):
        
        strides =  lags[i]//lags[i-1]
        
        conv = Conv1D(filters = dense_nodes[i], kernel_size = strides, strides = 1, \
               activation = 'sigmoid', use_bias = True, padding = 'same')(dense_layers[i-1])
        
        pool = MaxPooling1D(pool_size = strides, strides = strides)(conv)
        
        dense_layers.append(pool)
        
    flattened = Flatten()(dense_layers[-1])
    final_layer = Dense(final_nodes, activation = 'sigmoid')(flattened)
    outputs = Dense(1)(final_layer)
    #outputs = concatenated
    
    model = Model(inputs = inputs, outputs = outputs)
    
    ad = optimizers.Adadelta(lr = 0.05)
    
    model.compile(loss = 'mse', optimizer = ad)
    #model.compile(loss = 'mse', optimizer = "sgd")
    
    return model

def LSTM_Ms_pool(lags, time_steps, processed_scales, dense_nodes, lstm_nodes, l2):

    number_layers = len(lags)
    
    # we get the max number of values required by the model
    
    max_input_values = np.max([lags[i]*time_steps[i] for i in range(number_layers)])
    
    inputs = Input(shape = (max_input_values, 1))
    
    dense_layers = []
    
    dense_layers.append(inputs)
    
    for i in range(1, number_layers):
        
        strides =  lags[i]//lags[i-1]
        
        conv = Conv1D(filters = dense_nodes[i], kernel_size = strides, strides = 1, \
               activation = 'sigmoid', use_bias = True, padding = 'same')(dense_layers[i-1])
        
        pool = MaxPooling1D(pool_size = strides, strides = strides)(conv)
        
        dense_layers.append(pool)
        
    lstm_layers = []
    
    for scale in processed_scales:
        
        lambda_layer = Lambda(z_n, arguments = {'position': time_steps[scale]})(dense_layers[scale])
        
        lstm = LSTM(lstm_nodes[scale], activation='sigmoid', recurrent_activation='sigmoid',\
                activity_regularizer=regularizers.l2(l2), \
                recurrent_regularizer=regularizers.l2(l2))(lambda_layer)
        
        lstm_layers.append(lstm)
        
    if len(lstm_layers) > 1:
        
        concatenated = keras.layers.concatenate(lstm_layers)
        
    else:
        
        concatenated = lstm_layers[0]
        
    outputs = Dense(1)(concatenated)
    #outputs = dense_layers[1]
    #outputs = concatenated
    model = Model(inputs = inputs, outputs = outputs)
    
    ad = optimizers.Adadelta(lr = 0.05)
    
    model.compile(loss = 'mse', optimizer = ad)
    #model.compile(loss = 'mse', optimizer = "sgd")
    
    return model


def TDNN(lags, dense_nodes, input_length, l2, final_nodes):

    number_layers = len(lags)
    
    # we get the max number of values required by the model
    
    #max_input_values = np.max([lags[i]*time_steps[i] for i in range(number_layers)])
    
    inputs = Input(shape = (input_length, 1))
    
    dense_layers = []
    
    dense_layers.append(inputs)
    
    for i in range(1, number_layers):
        
        strides =  lags[i]//lags[i-1]
        
        conv = Conv1D(filters = dense_nodes[i], kernel_size = strides, strides = strides, \
               activation = 'sigmoid', use_bias = True)(dense_layers[i-1])
        
        #pool = MaxPooling1D(pool_size = strides, strides = strides)(conv)
        
        dense_layers.append(conv)
        
    flattened = Flatten()(dense_layers[-1])
    final_layer = Dense(final_nodes, activation = 'sigmoid')(flattened)
    outputs = Dense(1)(final_layer)
    #outputs = concatenated
    
    model = Model(inputs = inputs, outputs = outputs)
    
    ad = optimizers.Adadelta(lr = 0.05)
    
    model.compile(loss = 'mse', optimizer = ad)
    #model.compile(loss = 'mse', optimizer = "sgd")
    
    return model
        
def test(): 
    
    lags = [1,24,48]
    processed_scales = [0]
    dense_nodes = [1, 5, 5]
    
    #time_steps = [1,2,4]
    lstm_nodes = [1, 1]
    l2 = 0.001
    values = int(48*10)
    mod = Conv(lags, dense_nodes,values, l2)
    
    
    inputs = np.random.normal(size=(2,values,1))
    outputs = np.random.normal(size=(1,1))
    #outputs[:,1] = mod.predict(inputs)[:,1]
    print(mod.predict(inputs).shape)
#    print(mod.layers)
#    #weights = mod.layers[4].get_weights()
#    #print(mod.layers[1].get_weights())
#    print("i")
#    print(mod.predict(inputs))
#    print(outputs)
#
#    mod.fit(inputs,outputs,epochs=10)
#    print("i")
#    print(mod.predict(inputs))
#    #weights2 = mod.layers[4].get_weights()
    #print(np.abs(weights[0] - weights2[0]))
    
def test2(): 
    
    lags = [1,2,4]
    processed_scales = [1]
    dense_nodes = [1,2,1]
    
    time_steps = [1, 5,7]
    lstm_nodes = [4, 3,5]
    l2 = 0.001
    #values = int(48*10)
    values = np.max([lags[i]*time_steps[i] for i in range(len(lags))])
    mod,mod2 = LSTM_Ms(lags, time_steps, processed_scales, dense_nodes, lstm_nodes, l2)
    
    
    inputs = np.ones((1,values,1))
    outputs = np.random.normal(size=(1,1))
    #outputs[:,1] = mod.predict(inputs)[:,1]
    #print(mod.predict(inputs).shape)
    return mod,mod2, inputs

        