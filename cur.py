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
    selected_col.append(random.randint(0,n_movies-1))	#Columns selected randomly
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
    selected_row.append(random.randint(0,n_users-1))	#Rows selected randomly
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
W_cur = np.diag(W_cur)
print(W_cur)
for i in range(W_cur.shape[0]):
    W_cur[i][i] = 1/W_cur[i][i]
U = np.dot(np.dot(W2.T, W_cur**2), W1.T)
cur_100 = np.dot(np.dot(C, U), R.T)

'''
All ratings estimated to be greater than 5 or less than 0 are rewritten
'''


for i in range(cur_100.shape[0]):
    for j in range(cur_100.shape[1]):
        if cur_100[i,j] > 5:
            cur_100[i,j] = 5
        elif cur_100[i,j] < 0:
            cur_100[i,j] = 0
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

print(RMSE(cur_100, matrix))
print(MAE(cur_100, matrix))
print(end_time - start_time)