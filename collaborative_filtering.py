import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from collections import Counter
from time import time

'''
	Pickle library is used to load train and test matrices, users_map and movies_map saved from preprocess.py
'''

filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
n_users = matrix.shape[0]
n_movies = matrix.shape[1]

filehandler = open("test_dump", 'rb+')
test = pickle.load(filehandler)

filehandler = open("user_dump", 'rb+')
users_map = pickle.load(filehandler)

filehandler = open("movie_dump", 'rb+')
movies_map = pickle.load(filehandler)


matrix = matrix.T	#The rows are now movies and columns are users
'''
The pearson co-efficient matrix between movies to find similarities between two movies
'''
sim_matrix = np.corrcoef(matrix)
'''
Using the test cases to estimate the ratings using collaborative filtering (item-item)
'''
actual_ratings = []
pred_ratings = []

start_time = time()

for i in range(len(test["movieid"])):
    user = test.iloc[i,0]
    movie = test.iloc[i,1]
    rating = test.iloc[i,2]
    movie = movies_map[str(movie)]
    user = users_map[str(user)]
    actual_ratings.append(int(rating))
    
    num = 0
    den = 0
    sim_movie = sim_matrix[movie]
    user_ratings = matrix[:,user]
    for j in range(n_movies):
        if user_ratings[j] != 0:
            num = num + sim_movie[j]*user_ratings[j]
            den = den + sim_movie[j]
    pred_ratings.append(num/den)
end_time = time()

def RMSE(pred,value):
    N = len(pred)
    sum = np.sum(np.square(pred-value))
    return np.sqrt(sum/N)
	
def MAE(pred,value):
    N = len(pred)
    sum = np.sum(abs(pred-value))
    return sum/N

print(RMSE(np.array(pred_ratings), np.array(actual_ratings)))
print(MAE(np.array(pred_ratings), np.array(actual_ratings)))
print(end_time - start_time)