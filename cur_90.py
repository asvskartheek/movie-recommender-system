'''
CUR Algorithm:
Sparse matrix A can be represented as C*U*R where C is a matrix consisting of columns of A and R is a matrix consisting of rows of A.
C and R are sparse matrices while U is dense.
'''
import numpy as np
import pickle
from collections import Counter
from numpy.linalg import svd
import numpy.linalg as linalg
from time import time
import random

'''
	Pickle library is used to load matrix, users_map and movies_map saved from preprocess.py
'''
filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
n_users = matrix.shape[0]
n_movies = matrix.shape[1]

filehandler = open("user_dump", 'rb+')
users_map = pickle.load(filehandler)

filehandler = open("movie_dump", 'rb+')
movies_map = pickle.load(filehandler)

start_time = time()

'''
The mean rating of each user is calculated
'''
users_mean = matrix.sum(axis=1)
counts = Counter(matrix.nonzero()[0])
for i in range(n_users):
    if i in counts.keys():
        users_mean[i] = users_mean[i]/counts[i]
    else:
        users_mean[i] = 0

'''
The mean rating of each movie is calculated
'''
movies_mean = matrix.T.sum(axis=1)
counts = Counter(matrix.T.nonzero()[0])
for i in range(n_movies):
    if i in counts.keys():
        movies_mean[i] = movies_mean[i]/counts[i]
    else:
        movies_mean[i] = 0

'''
The probabilities of selection along the columns and rows is calculated
'''
total_norm =  np.linalg.norm(matrix)
col_norm =  np.linalg.norm(matrix,axis = 0)
row_norm =  np.linalg.norm(matrix,axis = 1)
for i in range(n_movies):
    col_norm[i] = (col_norm[i]/total_norm)**2
    
for i in range(n_users):
    row_norm[i] = (row_norm[i]/total_norm)**2

'''
Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
'''
c=800
selected_col = []
C = np.zeros([n_users,c])
for i in range(c):
    selected_col.append(random.randint(0,n_movies-1))
i=0
duplicate = len(selected_col) - len(set(selected_col))
for x in selected_col:
    p = col_norm[x]
    d = np.sqrt(c*p)
    if duplicate == 0:
        C[:,i] = matrix[:,x]/d
    else:
        C[:,i] = (matrix[:,x]/d)*(duplicate)**0.5
    i = i+1

'''
Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
'''
r=800
selected_row = []
R = np.zeros([n_movies,r])
for i in range(r):
    selected_row.append(random.randint(0,n_users-1))
i=0
duplicate = len(selected_row) - len(set(selected_row))
for x in selected_row:
    p = row_norm[x]
    d = np.sqrt(r*p)
    if duplicate == 0:
        R[:,i] = matrix.T[:,x]/d
    else:
        R[:,i] = (matrix.T[:,x]/d)*(duplicate)**0.5
    i = i+1

'''
The matrix U is constructed from W by the Moore-Penrose pseudoinverse
This step involves using SVD to find U and V' of W.
W is calculated as the intersection of the selected rows and columns
'''
W = C[selected_row,:]
W1, W_cur, W2 = svd(W)

'''
Dimensionality reduction in CUR by removing elements in diagonal of W_cur till square of sum of the diagonals is 90% of the original square sum
'''
W_cur = np.diag(W_cur)
total_sum = 0
dimensions = W_cur.shape[0]
for i in range(dimensions):
    total_sum = total_sum + np.square(W_cur[i,i])	#Find square of sum of all diagonals
retained = total_sum
while dimensions > 0:
	retained = retained - np.square(W_cur[dimensions-1,dimensions-1])
	if retained/total_sum < 0.9:		#90% energy retention
		break
	else:
		W1 = W1[:,:-1:]
		W2 = W2[:-1,:]
		W_cur = W_cur[:,:-1]
		W_cur = W_cur[:-1,:]
		dimensions = dimensions - 1		#Dimensionality reduction

for i in range(W_cur.shape[0]):
    W_cur[i][i] = 1/W_cur[i][i]
U = np.dot(np.dot(W2.T, W_cur**2), W1.T)
cur_90 = np.dot(np.dot(C, U), R.T)		#A = C*U*R

'''
All ratings estimated to be greater than 5 or less than 0 are rewritten
'''
for i in range(cur_90.shape[0]):
    for j in range(cur_90.shape[1]):
        cur_90[i,j] = cur_90[i,j]*10
        if cur_90[i,j] > 5:
            cur_90[i,j] = 5
        elif cur_90[i,j] < 0:
            cur_90[i,j] = 0
end_time = time()
			
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

print(RMSE(cur_90, matrix))
print(MAE(cur_90, matrix))
print(end_time - start_time)