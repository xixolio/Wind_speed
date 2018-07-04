# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:41:37 2018

@author: iaaraya
"""

from itertools import product
import subprocess
import sys


model = sys.argv[1]

experiment = int(sys.argv[2])

#model = "hierarchical_LSTM"
#experiment = 1

if model == "simple_LSTM":
    
    if experiment == 1:
        
        layers = ['[5-5]']
        
        lag = [1]
        
        time_steps = [5,10,15]
        
        epochs = [2]
        
        l2 = [0.001]
        
        learning_rate = [0.05]
        
        
        batch_size = [1]
        
        verbose = [1]
        
        
    if experiment == 2:
        
        layers = ['[5-5-5]']
        
        lag = [1]
        
        time_steps = [5,10,15]
        
        epochs = [2]
        
        l2 = [0.001]
        
        learning_rate = [0.05]
        
        batch_size = [1]
        
        verbose = [1]
        
    if experiment == 3:
        
        layers = ['[5]']
        
        lag = [1]
        
        time_steps = [5,10,15]
        
        epochs = [2]
        
        l2 = [0.001]
        
        learning_rate = [0.05]
        
        batch_size = [1]
        
        verbose = [1]
        
    combs = product(layers, lag, time_steps, epochs, l2, learning_rate, batch_size, verbose)
    
    string = ''
    
    for c in combs:
        
        if c:
            
            
            for element in c:
                
                string += str(element) + ','
                
            string += "--"
          
    model = "simple_LSTM"
    path = "/home/iaraya/Wind_speed/"
    name = "no_mvs_b08.csv"
    
    
    #print(string)
            
    subprocess.call(["python","main_gpu.py", model, path, name, string])



            
    
elif model == "hierarchical_LSTM":
    
      
    if experiment == 1:
                
        lags = ["[1-24]"]
        
        time_steps = ["[5-5]","[10-10]","[15-15]"]
        
        dense_nodes = ["[5-5]"]
        
        lstm_nodes = ["[5-5]"]
        
        #lstm_nodes = ["[10-10]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [5]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
    
        
    if experiment == 3:
        
        lags = ["[1-24-48]"]
        
        #time_steps = ["[24-1-1]","[24-5-1]","[24-10-1]","[24-15-1]", \
        #              "[24-15-1]","[24-15-5]","[24-15-15]"]
        
        #time_steps = ["[20-20-20]"]
        time_steps = ["[5-5-5]","[10-10-10]","[15-15-15]"]
        #dense_nodes = ["[1-10-10]","[1-5-5]"]
        
        dense_nodes = ["[5-5-5]"]
        
        #lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        lstm_nodes = ["[5-5-5]"]
        
        processed_scales = ["[0-1-2]"]
        
        epochs = [5]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [1]
    
    verbose = [1]
    
    
    combs = product(lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                    epochs, l2, batch_size, shuffle, verbose)
    
    string = ''
    
    for c in combs:
        
        if c:
            
            
            for element in c:
                
                string += str(element) + ','
                
            string += "--"
          
    model = "hierarchical_LSTM"
    path = "/home/iaraya/Wind_speed/"
    name = "no_mvs_b08.csv"
    
    
    #print(string)
            
    subprocess.call(["python","main_gpu.py", model, path, name, string])





