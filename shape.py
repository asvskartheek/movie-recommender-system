import pickle
import numpy

filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
print(matrix.shape)