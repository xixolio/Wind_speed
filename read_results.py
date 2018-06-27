# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:25:42 2018

@author: iaaraya
"""

import numpy as np    
import sys


# Simple_LSTM

file_name = "no_mvs_e01.csv"
my_file = "simple_LSTM_test_" + file_name[:-4] + ".txt"

my_file = "simple_LSTM_"+ file_name[:-4] + ".txt"
data_mae = []
data_mse = []

for i in range(10):

    f = open("results/" + str(i) + my_file)
    
    lines = f.readlines()[2:]
    f.close()
    
    data_mae_set = []
    data_mse_set = []
    
    for line in lines:
        
        data = line.split(' ')
        data_mae_set.append(float(data[-7]))
        data_mse_set.append(float(data[-5]))
        
    data_mae.append(data_mae_set)
    data_mse.append(data_mse_set)
    
data_mae = np.array(data_mae)
data_mse = np.array(data_mse)

mean_simple_LSTM_mae = np.mean(data_mae,0)
mean_simple_LSTM_mse = np.mean(data_mse,0)
#%%

# persistence

file_name = "no_mvs_villa_tehuelches.csv"
my_file = "persistence_" + file_name[:-4] + ".txt"
data_mae = []
data_mse = []

for i in range(10):

    f = open("results/" + str(i) + my_file)
    lines = f.readlines()
    f.close()
    
    data_mae_set = []
    data_mse_set = []
    
    for line in lines:
        
        data = line.split(' ')
        data_mae_set.append(float(data[-5]))
        data_mse_set.append(float(data[-4]))
        
    data_mae.append(data_mae_set)
    data_mse.append(data_mse_set)
    
data_mae = np.array(data_mae)
data_mse = np.array(data_mse)

mean_persistence_mae = np.mean(data_mae,0)
mean_persistence_mse = np.mean(data_mse,0)
        
    
        
    


    