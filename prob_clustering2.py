# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:17:58 2017

@author: andreas
"""
import pandas as pd
import numpy as np
import scipy as sp
from random import randint
from sklearn import linear_model, datasets
import sys

n_clusters = 4
n_iter = 100
n_runs = 1
#rthresh =  np.random.rand(15)
#x = ((np.random.rand(100,15)) > rthresh[None,:])*1
x = (digits > 0) * 1
#zv = np.where(np.sum(x,axis=0) == 0)
#x = np.delete(x,zv,1)
#print(len(zv[0])-1,' zero-sum features removed')

#det = np.linalg.det(np.matrix(x) * np.matrix(np.transpose(x)))

#if (det <= 0):
    
#    print('non-positive determinant - there might be cases with high covariance')

dims = np.shape(x)

n_examples = dims[0]

n_feats=dims[1]

entropy_class = np.zeros(n_runs)
final_cl = np.zeros([n_examples,n_runs])

#p_xy = np.copy(initvals)

for i_run in range(n_runs):
    
    p_xy = np.random.rand(n_clusters,n_feats)
    
    J = np.zeros([n_examples,n_clusters])
    
    cl = np.zeros(n_examples)
    
    p_x = np.sum(x,axis=0)/n_examples
        
    J_ = np.zeros([n_examples,n_clusters])
    
    for i_iter in range(n_iter):
            
        J = np.transpose(np.linalg.norm((x - p_xy[:,None]),axis=2))
        
        diff = np.sum(np.abs(J - J_))
        
        if diff < 1e-2:
    
            break
            
        cl = np.zeros([n_examples,n_clusters])
        
        cl[range(0,n_examples),np.argmin(J,axis=1)] = 1
     
        cnts = np.sum(cl,axis=0)
    
        p_cl = cnts / n_examples
        
        p_xy = np.zeros(np.shape(p_xy))
    
        p_xy = np.array((np.matrix(np.transpose(cl)) * np.matrix(x))/cnts[:,None])
    
        J_ = np.copy(J)    
        
        sys.stdout.write("{0}>".format("--"))
        
        sys.stdout.flush()
    
    sys.stdout.write("\rIterations completed") 
    
    pred_cl = np.zeros([n_examples,n_clusters])
    sys.stdout.write("\rComputing mean entropy across conditionals..") 
    
    for i_x in range(0,n_examples):
                       
        likelihood_probs = np.transpose(p_xy[:,x[i_x,:]==1])#/np.sum((p_xy[:,x[i_x,:]==1]),axis=1)
                        
        likelihood = np.prod(likelihood_probs,axis=0)
            
        marginal = (p_x[x[i_x,:]==1])
        
        marginal = np.prod(p_x[x[i_x,:]==1])
                    
        pred_cl[i_x,:] = (p_cl * likelihood) / marginal
    
    sum_pred_cl = np.sum(pred_cl,axis=1)
    
    pred_cl = np.nan_to_num(pred_cl/sum_pred_cl[:,None])
    
    A=np.matrix(np.nan_to_num(np.log(pred_cl))) * np.matrix(np.transpose(pred_cl))
    
    entropy_class[i_run] = -np.mean(np.diag(A))
    
    sys.stdout.write("Done.\r")
    
    cl = np.array((np.matrix(cl) * (np.transpose(np.matrix([range(n_clusters)])))))
    
    final_cl[:,i_run] = cl[:,0]
    
    print(entropy_class[i_run])
    
print(np.argmin(entropy_class))
    

