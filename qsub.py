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
#model = "simple_LSTM"

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
            
    
    




