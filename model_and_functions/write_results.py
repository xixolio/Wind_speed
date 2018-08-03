# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:53:07 2018

@author: iaaraya
"""
import os

def write_results(path,name,params,mae,mse,runs):
               
    my_file = path+name
    
    if not os.path.exists(my_file):
    #if not my_file.is_file():
        
        f = open(path + name, "a")

    else:
        
        f = open(path + name, "a")
            

    f.write('{}; {}; {} \n'.format(', '.join(str(x) for x in params) \
            , ', '.join(str(x) for x in mae.flatten()),', '.join(str(x) for x in mse.flatten())))
    
    f.close()