# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:13:24 2016

@author: henry
"""

import numpy as np
import pandas as pd

def JLT(data,m):
    data=np.matrix(data)    
    result=[]
    #construct JLT matrix
    A=np.matrix(np.random.randn(m,data.shape[1]))
    #transform the data
    transformed_data=data.dot(A.transpose())/np.sqrt(m)
    return pd.DataFrame(transformed_data)

#construct data matrix
size=100
n=20
data = np.matrix(np.random.randn(size,n))
print(JLT(data,10))
    
