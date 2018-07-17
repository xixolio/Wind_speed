
from itertools import product
import subprocess
import sys


model = sys.argv[1]

experiment = int(sys.argv[2])

#model = "hierarchical_LSTM"
#experiment = 2

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
        
    if experiment == 0:
                
        lags = ["[1-24]"]
        
        time_steps = ["[24-10]"]
        
        dense_nodes = ["[1-5]"]
        
        lstm_nodes = ["10-10]"]
        
        #lstm_nodes = ["[10-10]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [10]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
      
    if experiment == 1:
                
        lags = ["[1-24]"]
        
        time_steps = ["[24-1]","[24-5]","[24-10]","[24-15]"]
        
        dense_nodes = ["[1-5]","[1-10]"]
        
        lstm_nodes = ["10-10]","[20-20]"]
        
        #lstm_nodes = ["[10-10]"]
        
        processed_scales = ["[0-1]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
        
    if experiment == 2:
                
        lags = ["[1-24]"]
        
        time_steps = ["[24-1]","[24-5]","[24-10]","[24-15]"]
        
        dense_nodes = ["[1-5]", "[1-10]"]
        
        #lstm_nodes = ["[20-20]","[30-30]"]
        
        lstm_nodes = ["[10-10]", "[20-20]"]
        
        processed_scales = ["[1]"]
        
        epochs = [10,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [1]
        
    if experiment == 3:
        
        lags = ["[1-24-48]"]
        
        time_steps = ["[24-1-1]","[24-5-1]","[24-10-1]","[24-15-1]", \
                      "[24-15-1]","[24-15-5]","[24-15-15]"]
        
        #time_steps = ["[24-10-1]"]
        
        #dense_nodes = ["[1-10-10]","[1-5-5]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]"]
        
        #lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        lstm_nodes = ["[10-10-10]", "[20-20-20]"]
        
        processed_scales = ["[0-1-2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [1]
        
    if experiment == 4:
                
        lags = ["[1-24-48]"]
        
        time_steps = ["[24-1-1]","[24-5-1]","[24-10-1]","[24-15-10]", \
                      "[24-15-1]","[24-15-5]","[24-15-15]"]
        #dense_nodes = ["[1-10-10]","[1-20-20]"]
        
        #time_steps = ["[24-5-1]"]
        
        dense_nodes = ["[1-5-5]", "[1-10-10]"]
        
        lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        #lstm_nodes = ["[20-20-20]"]
        
        processed_scales = ["[1-2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [1]
        
    if experiment == 5:
                
        lags = ["[1-24-48]"]
        
        time_steps = ["[24-5-1]",\
                      "[24-20-5]","[24-20-10]","[24-20-15]"]
        
        time_steps = ["[24-20-15]"]
        
        dense_nodes = ["[1-10-10]","[1-5-5]"]
        
        #dense_nodes = ["[1-5-5]"]
        
        lstm_nodes = ["[10-10-10]","[20-20-20]"]
        
        processed_scales = ["[2]"]
        
        epochs = [1,5,10,15,20]
        
        l2 = [0.001]
        
        batch_size = [1]
        
        shuffle = [0]
        
        verbose = [0]
    
    verbose = [1]
    #epochs = [10,20]
    
    combs = product(lags, time_steps, dense_nodes, lstm_nodes, processed_scales,\
                    epochs, l2)
    
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
#            string = 'hierarchical_LSTM /user/i/iaraya/Wind_speed/data/  \
#                    no_mvs_e08.csv ' + string
#            
#            #print(string)
#            
#            subprocess.call(["qsub","main.sh","-F",string])
            
    for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ','
            
            model = 'hierarchical_LSTM'
            #path = '/user/i/iaraya/Wind_speed/data/'
            path = '/home/iaraya/CIARP/Wind_speed/data/'
            file = 'no_mvs_e01.csv'
            #string = 'hierarchical_LSTM /user/i/iaraya/Wind_speed/data/  \
            #        no_mvs_e08.csv ' + string
            
            #print(string)
            
            subprocess.call(["python","main.py",model, path, file,string])





