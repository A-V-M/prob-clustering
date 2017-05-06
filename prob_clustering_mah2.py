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

n_clusters = 2
n_iter = 1000
n_runs = 1
rthresh =  np.random.rand(50)
x = ((np.random.rand(750,50)) > 0.6)*1
#x = (digits > 0) * 1
#x = X
zv = np.where(np.sum(x,axis=0) == 0)
x = np.delete(x,zv,1)
print(len(zv[0]),' zero-sum features removed')

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
    
    VI = np.zeros((n_feats,n_feats,n_clusters))
    
    for i_clust in range(n_clusters):
        
        VI[:,:,i_clust] =  np.identity(n_feats)
    
    J = np.zeros([n_examples,n_clusters])
    
    cl = np.zeros(n_examples)
    
    p_x = np.sum(x,axis=0)/n_examples
        
    J_ = np.zeros([n_examples,n_clusters])
    
    for i_iter in range(n_iter):
        
        for i_clust in range(n_clusters):
            
            for i_example in range(n_examples):
                
                v = x[i_example,:] - p_xy[i_clust,:]
                J[i_example,i_clust] = np.dot(np.dot(v,VI[:,:,i_clust]),v)        

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
        
        
        for i_clust in range(n_clusters):
            
    
             rank = np.linalg.matrix_rank(np.cov(np.transpose(x[cl[:,i_clust]==1,:])))
             
             if rank == n_feats:
            
                 VI[:,:,i_clust]=np.linalg.inv(np.cov(np.transpose(x[cl[:,i_clust]==1,:])))

             else:
                 
                 print('Co-variance matrix is singular. Going Euclidean...')
#            VI[:,:,i_clust]=(np.dot(np.transpose(x[cl[:,i_clust]==1,:]),x[cl[:,i_clust]==1,:]))
              
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
    
    cl0 = np.array((np.matrix(cl) * (np.transpose(np.matrix([range(n_clusters)])))))
    
    final_cl[:,i_run] = cl0[:,0]
    
    print(entropy_class[i_run])
    
print(np.argmin(entropy_class))
    

