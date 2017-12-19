#!/bin/python
#cython: boundscheck=False
import numpy as np
cimport numpy as np
# from Inquiry.network cimport *           
# import itertools as it

DTYPE = np.int
ctypedef np.int_t DTYPE_t

np.random.seed(1)

"""" This file contains utility functions for neural networks """

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

def one_hot_encode(np.ndarray vector): 
    """ One Hot Encoding of a whole number valued rectangular vector with entries less than 2^16.  """

    if vector.shape[1] != 1: 
        vector = vector.flatten()
        
    num_classes = np.unique(vector).shape[0]
    cdef int minimum = np.min(vector)
    cdef int maximum = np.max(vector)

    one_hot_vector = np.zeros((vector.shape[0], num_classes), dtype='intc') # Get the matrix size correct  
    cdef int i,j
    cdef int c = 0 

    for j in np.nditer(vector):
        for i in range(minimum, maximum + 1):
            if i == j:
                one_hot_vector[c, i - minimum] = 1 
        c += 1
        
    return one_hot_vector 

def zero_vector_p(double[:,:] vector):
    """ Returns True if the vector is the zero vector """
    return False if np.count_nonzero(vector) else True

def one_hot_decode(long[:,:] matrix):
    """ Takes a one hot encoded numpy array and returns a vector of the classes """
    cdef long[:] decoded_vector # Everything else is doubles, but python is passing longs. What to do?
    cdef long[:] vector
    cdef unsigned int length_of_vector
    
    if np.ndim(matrix) != 1:
        for vector in matrix:
            length_of_vector = np.argmax(vector)
            
        decoded_vector = np.array([np.argmax(vector) for vector in matrix])

    else:
        decoded_vector = np.array(np.argmax(matrix))

    return decoded_vector 

def round_to_int(np.ndarray x):
    """ Rounds the array, and returns it with entries as numpy integers."""
    x = np.array(x)
    return np.round(x).astype(int)

# def load_network(char[:] file_name, **kwargs):
#     """ Load a network from file *file_name* """
#     n = NeuralNetwork()
#     weights = np.load(file_name, **kwargs)
#     n.weights = weights
#     return n 
    
