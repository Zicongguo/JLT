# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:13:24 2016

@author: henry
"""

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

def JLT(data,m,mode='classic'):
    valid = ['classic','toeplitz']
    if mode not in valid:
        raise ValueError("results: status must be one of %r." % valid)
    data=np.matrix(data)
    #construct JLT matrix
    if mode=='classic':  
        #A is a matrix with independent standard normal entries
        A=np.matrix(np.random.randn(m,data.shape[1]))
    elif mode=='toeplitz':
        col=np.random.randn(1,m)
        row=np.random.randn(1,data.shape[1])
        #A is a toeplitz matrix with independent standard normal entries in the first column and row
        P=toeplitz(col,row)
        #diagonal matrix with diagonal entries sampled from +-1 with equal probability
        D=np.diag(2*np.random.binomial(1,0.5,data.shape[1])-1)
        A=P.dot(D)
    #transform the data
    transformed_data=data.dot(A.transpose())/np.sqrt(m)
    return pd.DataFrame(transformed_data)



#test the function above
size=100
n=20
data = np.matrix(np.random.randn(size,n))
print(JLT(data,10,mode='toeplitz'))
    
