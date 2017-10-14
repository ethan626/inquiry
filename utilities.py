#!/bin/python
import numpy as np
import itertools
import json 

np.random.seed(1)

"""" This file contains utility functions for neural networks """

def one_hot_encode(vector, skip_null=False):
    """ One Hot Encoding of a whole number valued vector. 
        If skip_null is true, a vector of all zeros is ignored
       """
    vector = np.array(vector)
    vector = vector.flatten()
    num_classes = np.unique(vector)
    minimum = min(vector)
    maximum = max(vector)

    if skip_null:
        return np.array([[1 if i == j else 0 for i in range(minimum, maximum + 1)] for j in vector])

    else:
        return np.array([[1 if i == j else 0 for i in range(minimum, maximum + 1)] for j in vector])

def zero_vector_p(vector):
    """ Returns True if the vector is the zero vector """
    return False if np.count_nonzero(vector) else True

def one_hot_decode(matrix):
    """ Takes a one hot encoded numpy array and returns a vector of the classes """
    
    if np.ndim(matrix) != 1:
        return np.array([np.argmax(vector) for vector in matrix])

    else:
        return np.array(np.argmax(matrix))

def round_to_int(x):
    """ Rounds the array, and returns it with entries as numpy integers """
    x = np.array(x)
    return np.round(x).astype(int)

def load_network(file_name, **kwargs):
    """ Load a network from file *file_name* """
    n = NeuralNetwork()
    weights = np.load(file_name, **kwargs)
    n.weights = weights
    return n 
    
