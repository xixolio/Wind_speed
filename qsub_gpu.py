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
        
        layers = ['[5]', '[10]', '[15]', '[20]']
        
        lag = [12, 24, 36]
        
        time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,\
                      18, 19, 20]
        
        epochs = [1, 5]
        
        l2 = [0.001]
        
        learning_rate = [0.05]
        
        
    if experiment == 2:
        
        layers = ['[5]']
        
        lag = [12]
        
        time_steps = [5]
        
        epochs = [1]
        
        l2 = [0.001]
        
        learning_rate = [0.05]
        
    combs = product(layers, lag, time_steps, epochs, l2, learning_rate)
    
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            string = 'simple_LSTM /user/i/iaraya/CIARP/Wind_speed/data/ \
                    no_mvs_d05a.csv ' + string
            
            subprocess.call(["qsub","main.sh","-F",string])
            
    
elif model == "hierarchical_LSTM":
    
      
    if experiment == 1:
        
        lags = ["[1-6]",["1-12"]]
        
        time_steps = ["[4-4]"]
        
        dense_nodes = ["[1-5]"]
        
        lstm_nodes = ["[6-6]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [1]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-6]"]
        
        time_steps = ["[4-4]"]
        
        dense_nodes = ["[1-5]"]
        
        lstm_nodes = ["[6-6]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [1]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [1]
    
    
    combs = product(lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                    epochs, l2, batch_size, shuffle, verbose)
    
    string = ''
    
    for c in combs:
        
        if c:
            
            
            for element in c:
                
                string += str(element) + ','
                
            string += ":"
          
    model = "hierarchical_LSTM"
    path = "/user/i/iaraya/CIARP/Wind_speed/"
    name = "no_mvs_d05a.csv"
    
    
    #print(string)
            
    subprocess.call(["python","main_gpu.py", model, path, name])





