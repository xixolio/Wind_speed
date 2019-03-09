# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:54:53 2018

@author: iaaraya
"""
import sys

def get_params(argv_position):
    
    params = str(sys.argv[argv_position]).split(',')
    
    layers = str(params[0]).strip('[]')
    layers = [int(layer) for layer in layers.split('-')]
    
    lag = int(params[1])
    time_steps = int(params[2])
    epochs = int(params[3])
    l2 = float(params[4])
    learning_rate = float(params[5])
    batch_size = int(params[6])
    
    return layers, lag, time_steps, epochs, l2, learning_rate, batch_size

def get_params_Ms(argv_position):
    
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
    
    batch_size = int(params[7])
        
    shuffle = bool(int(params[8]))
    
    final_nodes = int(params[9])
    
    return lags, time_steps, dense_nodes, lstm_nodes, processed_scales, \
            epochs, l2, batch_size, shuffle, final_nodes
            
def get_params_Conv(argv_position):
    
    params = str(sys.argv[argv_position]).split(',')

    lags = str(params[0]).strip('[]')
    lags = [int(lag) for lag in lags.split('-')]
    
    dense_nodes = str(params[1]).strip('[]')
    dense_nodes = [int(dn) for dn in dense_nodes.split('-')]
    
    input_length = int(params[2])
    
    final_nodes = int(params[3])
    
    epochs = int(params[4])
    
    l2 = float(params[5])
    
    batch_size = int(params[6])
        
    shuffle = bool(int(params[7]))
    
    return lags, dense_nodes, input_length, final_nodes, epochs, l2,\
            batch_size, shuffle