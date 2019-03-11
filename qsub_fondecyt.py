# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:11:54 2019

@author: iaaraya
"""


from itertools import product
import subprocess
import sys
import numpy as np


model = sys.argv[1]

experiment = int(sys.argv[2])

setting = sys.argv[3]

path = sys.argv[4]

file = sys.argv[5]

test = sys.argv[6]

#model = "hierarchical_LSTM"
#experiment = 2

if model == "simple_LSTM":
    
    if experiment == 0:
        
        layers = ['[5]']
        lag = [24]
        time_steps = [12]
        epochs = [100]
        l2 = [0.001]
        learning_rate = [0.001]
        batch_size = [32]
        
    if experiment == 1:
        
        layers = ['[5]', '[10]', '[15]', '[20]','[25]','[30]']
        
        lag = [1,24]
        
        time_steps = [1,10,20,30,40,50]
        
        #epochs = [10, 20]
        
        l2 = [0]
        
        learning_rate = [0.1]
        
        
    if experiment == 2:
        
        layers = ['[5-5]', '[10-10]', '[15-15]', '[20-20]','[25-25]','[30-30]']
        
        lag = [1,24]
        
        time_steps = [1,10,20,30,40,50]
        
        epochs = [1]
        
        l2 = [0]
        
        learning_rate = [0.1]
        
    batch_size = [32]
    epochs = [100]
    combs = product(layers, lag, time_steps, epochs, l2, learning_rate, batch_size)
    
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            if setting == "fondecyt":
                subprocess.call(["python","main.py",model, path, file,string])
            elif setting == "cluster":
                string = str(model) +" "+path+" "+file+" "+string  
                subprocess.call(["qsub","main.sh","-F",string])
            
            
elif model == "LSTM_Ms" or model == "LSTM_Ms_pool" or model == "LSTM_Ms_locally" or model == 'LSTM_Ms_return' \
or model == "SRNN_Ms_return" and test=='test':  
    
    files = ['no_mvs_b08.csv','no_mvs_e01.csv','no_mvs_originald08.csv','no_mvs_d05a.csv']
    for file in files:
        for i in range(5):
            file_name = str(model) + '_' + file[:-4] + "set_"+str(i)+".txt"
            with open('best_val_results/best_'+file_name,'r') as file2:
                
                for line in file2.readlines():
                    string = line
                    if setting == "fondecyt":
                                subprocess.call(["python","main.py",model, path, file,string,'test',str(i)])
                    elif setting == "cluster":
                                string = string.replace('\n','')
                                print(string)
                                string = str(model) +" "+path+" "+file+" "+string+" test "+str(i) 
                                print(string)
                                subprocess.call(["qsub","main.sh","-F",string])
    

elif model == "LSTM_Ms" or model == "LSTM_Ms_pool" or model == "LSTM_Ms_locally" or model == 'LSTM_Ms_return' \
or model == "SRNN_Ms_return" and test=='validation':
        
    if experiment == 0:
                
        lags = ["[1-24]"]
        
        time_steps = ["[1-1]"]
        
        dense_nodes = ["[1-5]"]
        
        lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        processed_scales = ["[0-1]"]
        
        final_nodes = [10]
        
        epochs = [30]
        
        l2 = [0.001]
        
        batch_size = [32]
        
        shuffle = [1]
        
        verbose = [1]
      
    if experiment == 1:
                
        lags = ["[1-24]"]
        
        steps = ["1","10","15"]
        steps_pairs = product(steps,steps)
        time_steps = []
        
        for element in steps_pairs:
            
            time_steps.append("[" + "-".join(element) + "]")
                          
        #print(time_steps)
        
        time_steps = ["[12-1]","[12-5]","[12-10]","[12-15]"]
        
        final_nodes = [0,10,20]
        
        dense_nodes = ["[1-5]","[1-10]"]
        
        lstm_nodes = ["[10-10]","[20-20]"]
        
        #lstm_nodes = ["[10-10]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [10,20]
        
        l2 = [0.001]
        
        batch_size = [32]
        
        shuffle = [1]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-24]"]
        
        time_steps = ["[12-5]","[12-10]","[12-15]"]
        
        dense_nodes = ["[1-5]","[1-10]","[1-15]"]
        
        #lstm_nodes = ["[20-20]","[30-30]"]
        
        lstm_nodes = [ "[10-10]","[20-20]"]
        
        final_nodes = [5,10,15]
        
        processed_scales = ["[1]"]
        
        epochs = [10,20,30]
        
        l2 = [0.001]
        
        batch_size = [32]
        
        shuffle = [1]
        
        verbose = [1]
        
    if experiment == 3:
        
        lags = ["[1-24-48]"]
        
        time_steps = ["[12-1-1]","[12-5-1]","[12-10-1]","[12-15-1]", \
                      "[12-15-1]","[12-15-5]","[12-15-10]","[12-15-15]"]
        
        #time_steps = ["[24-10-1]"]
        
        #dense_nodes = ["[1-10-10]","[1-5-5]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]"]
        
        #lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        lstm_nodes = ["[10-10-10]", "[20-20-20]"]
        
        final_nodes = [5,10,15]
        
        processed_scales = ["[0-1-2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [32]
        
        shuffle = [0]
        
        verbose = [1]
        
    if experiment == 4:
                
        lags = ["[1-24-48]"]
        
        
        time_steps = ["[12-1-1]","[12-5-1]","[12-10-1]","[12-15-1]", \
                      "[12-15-1]","[12-15-5]","[12-15-10]","[12-15-15]"]
        
        
        #dense_nodes = ["[1-10-10]","[1-20-20]"]
        
        #time_steps = ["[24-5-1]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]"]
        
        lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        final_nodes = [5,10,15]
        
        #lstm_nodes = ["[20-20-20]"]
        
        processed_scales = ["[1-2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [32]
        
        shuffle = [0]
        
        verbose = [1]
        
    if experiment == 5:
                
        lags = ["[1-24-48]"]
        
        time_steps = ["[24-5-1]",\
                      "[24-20-5]","[24-20-10]","[24-20-15]"]
        
        #time_steps = ["[24-20-15]"]
        
        dense_nodes = ["[1-10-10]","[1-5-5]"]
        
        #dense_nodes = ["[1-5-5]"]
        
        lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        processed_scales = ["[2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
    
    #verbose = [1]
    shuffle = [1]
    epochs = [100]
    batch_size =[32]
    final_nodes = [0,10,20]
    
    combs = product(lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                    epochs, l2, batch_size, shuffle, final_nodes) 
    
#    for c in combs:
#        
#        if c:
#            
#            string = ''
#            
#            for element in c:
#                
#                string += str(element) + ','
#            
#            string = str(model) + ' /user/i/iaraya/Wind_speed/data/  \
#                    no_mvs_e01.csv ' + string
            
            #print(string)
            
            #subprocess.call(["qsub","main.sh","-F",string])

            
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            

            #path = '/home/iaraya/CIARP/'
            #file = 'no_mvs_e01.csv'

            if setting == "fondecyt":
                subprocess.call(["python","main.py",model, path, file,string])
            elif setting == "cluster":
                string = str(model) +" "+path+" "+file+" "+string  
                subprocess.call(["qsub","main.sh","-F",string])

                


elif model == "Conv":
    
    if experiment == 0:
        
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-15]"]
        
        multipliers = np.array([15])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        

    if experiment == 1:
                
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-5]", "[1-10]","[1-15]","[1-20]"]
        
        multipliers = np.array([1, 5, 10, 15, 20])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15,20]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-24-48]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]","[1-15-15]","[1-20-20]"]
        
        multipliers = np.array([1, 5, 10])
        
        input_length = 48*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15,20]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [10, 20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    shuffle = [1]
    epochs = [1000]
    batch_size =[32]
    final_nodes = [10,15,20]
    combs = product(lags, dense_nodes, input_length, final_nodes, epochs,\
                    l2, batch_size, shuffle)
    
    
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            if setting == "fondecyt":
                subprocess.call(["python","main.py",model, path, file,string])
            elif setting == "cluster":
                string = str(model) +" "+path+" "+file+" "+string  
                subprocess.call(["qsub","main.sh","-F",string])
            
elif model == "TDNN":
    
    if experiment == 0:
        
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-15]"]
        
        multipliers = np.array([15])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        

    if experiment == 1:
                
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-5]", "[1-10]","[1-15]"]
        
        multipliers = np.array([1, 5, 10, 15])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [10,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-24-48]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]","[1-15-15]"]
        
        multipliers = np.array([1, 5, 10, 15])
        
        input_length = 48*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [10, 20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    combs = product(lags, dense_nodes, input_length, final_nodes, epochs,\
                    l2, batch_size, shuffle)
    
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            string = 'TDNN /user/i/iaraya/Wind_speed/data/  \
                    no_mvs_e01.csv ' + string
            
            #print(string)
            
            subprocess.call(["qsub","main.sh","-F",string])
            
elif model == "TDNN_l":
    
    if experiment == 0:
        
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-15]"]
        
        multipliers = np.array([15])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        

    if experiment == 1:
                
        lags = ["[1-24]"]
        
        dense_nodes = ["[1-5]", "[1-10]","[1-15]"]
        
        multipliers = np.array([1, 5, 10, 15])
        
        input_length = 24*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [10,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-24-48]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]","[1-15-15]"]
        
        multipliers = np.array([1, 5, 10, 15])
        
        input_length = 48*multipliers
        input_length = input_length.tolist()
        
        final_nodes = [5,10,15]
        #lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        #processed_scales = ["[0-1]"]
        
        epochs = [10, 20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [1]
        
        verbose = [0]
        
    combs = product(lags, dense_nodes, input_length, final_nodes, epochs,\
                    l2, batch_size, shuffle)
    
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            string = 'TDNN_l /user/i/iaraya/Wind_speed/data/  \
                    no_mvs_e01.csv ' + string
            
            #print(string)
            
            subprocess.call(["qsub","main.sh","-F",string])


