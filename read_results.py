# -*- coding: utf-8 -*-

import numpy as np    
import sys


file_name = sys.argv[1]
my_file = "simple_LSTM_test_" + file_name[:-4] + ".txt"
data = []


for i in range(10):

    f = open("results/" + str(i) + my_file)
    
    lines = f.readlines()
    f.close()
    
    lines = lines[1]
    
    data = lines.split("[")[1:]
    
    corrected_data = []
    
    for data_line in data:
        
        corrected_data.append( "[" + data_line )
        
     
    #print(lines)

    