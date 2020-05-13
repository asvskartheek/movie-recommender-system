import numpy as np
import pickle
from collections import Counter
from numpy.linalg import svd
import numpy.linalg as linalg
from time import time
import random

filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
n_users = matrix.shape[0]
n_movies = matrix.shape[1]
S = np.zeros((n_users,n_movies),dtype=float)
Q,Sigma,V = svd(matrix)
Sigma = np.diag(Sigma)

S[:n_movies,:n_movies] = Sigma
print(S.shape)
P = np.dot(S,V)
#P = P.T
S = S[:7,:7]
Q = Q[:,:7]
P = P[:7,:]
print(Q.shape)
print(P.shape)
for k in range(15):
	for i in range(n_movies):
		for j in range(n_users):
			residual = matrix[j][i] - np.dot(Q[j,:],P[:,i])
			temp = P[:,i]
	        # we want to update them at the same time, so we make a temporary variable. 
			P[:,i] +=  (0.0001) * residual * Q[j,:]
			Q[j,:] +=  (0.0001) * residual * temp
	print(k)

def RMSE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    cur_sum = np.sum(np.square(pred-value))
    return np.sqrt(cur_sum/(N*M))

curr_matrix = np.dot(Q,P)
print(RMSE(curr_matrix,matrix))