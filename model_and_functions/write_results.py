# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:53:07 2018

@author: iaaraya
"""
import os
from functools import reduce

def write_results(path,name,params,mae,mse,h_mae,h_mse):
               
    #my_file = path+name
    
    #if not os.path.exists(my_file):
    #if not my_file.is_file():
        
    #f = open(path + name, "a")

    #else:
        
    #    f = open(path + name, "a")
            

    #f.write('{}; {}; {} \n'.format(', '.join(str(x) for x in params) \
    #        , ', '.join(str(x) for x in mae.flatten()),', '.join(str(x) for x in mse.flatten())))
    #print(path+name)
    with open(path+name,"a") as file:
        
        concat = lambda s: lambda x,y: str(x)+s+str(y)
        
        h_mae_str = reduce(concat(","),[reduce(concat(";"),[reduce(concat("-"),m) for m in sr]) for sr in h_mae])
        h_mse_str = reduce(concat(","),[reduce(concat(";"),[reduce(concat("-"),m) for m in sr]) for sr in h_mse])
        
        mae_str = reduce(concat(","),[reduce(concat(";"),sr) for sr in mae])
        mse_str = reduce(concat(","),[reduce(concat(";"),sr) for sr in mse])
        
        params_str = reduce(concat(","),params)
        
        results = [params_str,mae_str,mse_str,h_mae_str,h_mse_str]
    
        file.write(reduce(concat(" "),results) + "\n")
        
def write_result(path,name,params,mae,mse,h_mae,h_mse, epoch):
    with open(path+name,"a") as file:
        concat = lambda s: lambda x,y: str(x)+s+str(y)
        
        h_mae_str = reduce(concat(","),[reduce(concat(";"),m) for m in h_mae])
        h_mse_str = reduce(concat(","),[reduce(concat(";"),m) for m in h_mse])
        #h_mse_str = reduce(concat(","),[reduce(concat(";"),[reduce(concat("-"),m) for m in sr]) for sr in h_mse])
        
        mae_str = reduce(concat(","),mae)
        mse_str = reduce(concat(","),mse)
        
        params_str = reduce(concat(","),params)
        
        results = [params_str,str(epoch),mae_str,mse_str,h_mae_str,h_mse_str]
    
        file.write(reduce(concat(" "),results) + "\n")