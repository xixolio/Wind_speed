# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:25:42 2018

@author: iaaraya
"""

import numpy as np    
import sys


# Simple_LSTM

file_name = "no_mvs_e01.csv"
my_file = "correctedsimple_LSTM_test_" + file_name[:-4] + ".txt"

my_file = "hierarchical_LSTM_"+ file_name[:-4] + ".txt"
data_mae = []
data_mse = []
data = []

for i in range(10):

    f = open("results/" + str(i) + my_file)
    
    lines = f.readlines()[1:]
    f.close()
    
    data_mae_set = []
    data_mse_set = []
    
    for line in lines:
        
        data = line.split(' ')
        data_mae_set.append(float(data[-7]))
        data_mse_set.append(float(data[-6]))
        
    data_mae.append(data_mae_set)
    data_mse.append(data_mse_set)
    
data_mae = np.array(data_mae)
data_mse = np.array(data_mse)

mean_simple_LSTM_mae = np.mean(data_mae,0)
mean_simple_LSTM_mse = np.mean(data_mse,0)
#%%

# persistence

file_name = "no_mvs_d05a.csv"
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
        
#%%

   
        
file_name = "no_mvs_e01.csv"
my_file = "correctedsimple_LSTM_test_" + file_name[:-4] + ".txt"

my_file = "final2_hierarchical_LSTM_"+ file_name[:-4] + ".txt"
data_mae = []
data_mse = []
data = []
f = open("results/" + my_file)
lines = f.readlines()[-3:]
f.close()
results = np.zeros((len(lines), 10, 5))

for k,line in zip(range(len(lines)),lines):
    
    data = line.split(';')[1].split(' ')[1:]
    
    for i in range(10): 
        
        for j in range(5):
            
            results[k,i,j] = float(data[i*5 + j].strip(','))
            
 
mean_by_run = np.mean(results,axis = 1)
total_mean = np.mean(mean_by_run, axis = 1)
#data_mae = np.array(data_mae)
#data_mse = np.array(data_mse)
#
#mean_simple_LSTM_mae = np.mean(data_mae,0)
#mean_simple_LSTM_mse = np.mean(data_mse,0)


    