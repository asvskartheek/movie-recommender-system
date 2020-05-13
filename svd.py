'''
Singular value decomposition with 100% energy retention and 90% energy retention
The sparse matrix A can be represented as:
A = U*Sigma*V'
'''

import numpy as np
import pickle
import numpy.linalg as linalg
from time import time

'''
	Pickle library is used to load the sparse matrix saved from preprocess.py
'''
filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
n_users = matrix.shape[0]
n_movies = matrix.shape[1]
start_time = time()

'''
Calculating U
U = [eigenvectors of A*A'] in descending order of it's eigenvalues
Dimensions of U = n_users x rank(A*A')
'''
m1 = np.dot(matrix, matrix.T)
eigenValues, eigenVectors = linalg.eig(m1)
eigenValues = eigenValues.real
eigenVectors = eigenVectors.real
eigen_map = dict()
for i in range(len(list(eigenValues))):
	eigenValues[i] = round(eigenValues[i], 4)
for i in range(len(eigenValues)):
    if eigenValues[i] != 0:
        eigen_map[eigenValues[i]] = eigenVectors[:, i]
eigenValues = sorted(list(eigen_map.keys()), reverse=True)
U = np.zeros_like(eigenVectors)
for i in range(len(eigenValues)):
    U[:,i] = eigen_map[eigenValues[i]]
U = U[:,:len(eigenValues)]
#print(U.shape)
'''
Calculating Sigma
Sigma = Diagonal matrix with diagonal elements in descending order of non-zero eigenvalues of U or V.
Dimensions of Sigma = rank(A*A') x rank(A*A')
'''
Sigma = np.diag([i**0.5 for i in eigenValues])
#print(Sigma.shape)
'''
Calculating V
V = [eigenvectors of A'*A] in descending order of it's eigenvalues
Dimensions of U = n_movies x rank(A'*A)
In the end V is represented as V' for simplicity in furthur operations
'''
m2 = np.dot(matrix.T, matrix)
eigenValues, eigenVectors = linalg.eig(m2)
eigenValues = eigenValues.real
eigenVectors = eigenVectors.real
for i in range(len(list(eigenValues))):
	eigenValues[i] = round(eigenValues[i], 4)
eigen_map = dict()
for i in range(len(eigenValues)):
    if eigenValues[i] != 0:
        eigen_map[eigenValues[i]] = eigenVectors[:, i]
eigenValues = sorted(list(eigen_map.keys()), reverse=True)
V = np.zeros_like(eigenVectors)
for i in range(len(eigenValues)):
    V[:,i] = eigen_map[eigenValues[i]]
V = V[:,:len(eigenValues)]
V = V.T
#print(V.shape)
mid_time_1 = time()
'''
SVD with 100% energy retention
'''
shape = Sigma.shape
svd_100 = np.dot(np.dot(U,Sigma),V)
svd_100_time = time()

'''
Dimensionality reduction of Sigma, U and V is done by removing the lower diagonal elements of Sigma till the sum of diagonal elements in Sigma is 90% of the original value
Dimension of Sigma decreases from 3663x3663 to 883x883
'''
total_sum = 0
dimensions = Sigma.shape[0]
for i in range(dimensions):
    total_sum = total_sum + np.square(Sigma[i,i])	#Find square of sum of all diagonals
retained = total_sum
while dimensions > 0:
	retained = retained - np.square(Sigma[dimensions-1,dimensions-1])
	if retained/total_sum < 0.9:	#90% energy retention
		break
	else:
		U = U[:,:-1:]
		V = V[:-1,:]
		Sigma = Sigma[:,:-1]
		Sigma = Sigma[:-1,:]
		dimensions = dimensions - 1	#Dimensionality reduction
mid_time_2 = time()

'''
SVD with 90% energy retention
'''
svd_90 = np.dot(np.dot(U,Sigma),V)
svd_90_time = time()

def RMSE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    cur_sum = np.sum(np.square(pred-value))
    return np.sqrt(cur_sum/(N*M))

def MAE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    sum = np.sum(abs(pred-value))
    return (sum)/(N*M)

print("SVD:")
print(RMSE(svd_100, matrix))
print(MAE(svd_100, matrix))
print(svd_100_time - start_time)
print("\nSVD with 90% energy:")
print(RMSE(svd_90, matrix))
print(MAE(svd_90, matrix))
print(svd_90_time - mid_time_2 + mid_time_1 - start_time)