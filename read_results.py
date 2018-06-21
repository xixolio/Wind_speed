# -*- coding: utf-8 -*-

import numpy as np    

my_file = "simple_LSTM_test" + file_name[-4:] + ".txt"
data = []


for i in range(10):
    
    f = open("results/"+str(i)+".txt")
    my_file = path+str(i)+name
    model_name = "persistence_" + str(i) + "_run"
    lines = f.readlines()
    f.close()
    
    init = 2
    final= 0
    lines = lines1[init:]
     
    errors2.append([float(line.split(' ')[-3]) for line in lines]) 
    errors3.append([float(line.split(' ')[-2]) for line in lines])
    errors4.append([float(line.split(' ')[-1]) for line in lines])

    data.append([line for line in lines])
    
min_index = np.argmin(errors2,axis=1)
data_min2 = []
for i in range(10):
    data_min2.append(data[i][min_index[i]])
    
errors = np.mean(errors2,axis=0)
errors_2 = np.mean(errors3,axis=0)
errors_3 = np.mean(errors4,axis=0)
print(np.min(errors),errors_2[np.argmin(errors)],errors_3[np.argmin(errors)])
mins = []
for i in range(10):
    mins.append(errors2[i][np.argmin(errors)])
    print("%.3f" % errors2[i][np.argmin(errors)] )