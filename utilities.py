#!/bin/python
import numpy as np
import itertools
np.random.seed(1)

"""" This file contains utility functions for neural networks """

def one_hot_encode(vector, skip_null=False):
    """ One Hot Encoding ofa positive integer valued vector. 
        Vectors of all zeros are of type -1. If skip_null is true, a vector of all zeros is ignored
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

def one_hot_decode(matrix):
    """ Takes a one hot encoded numpy array and returns a vector of the classes """
    def zero_vector_p(vector):
        """ Returns True if the vector is the zero vector """
        return False if np.count_nonzero(vector) else True
    
    def decode(vector):
        """ One hot decodes the individual vectors of a matrix. If no type can be infered, ie the zero vector, 1 is assinged as the type"""
        if zero_vector_p(vector):
            return -1
        else:
            for index, element in enumerate(vector):
                if element == 1:
                    return index 

    if np.ndim(matrix) != 1:
        return np.array([decode(vector) for vector in matrix])

    else:
        return np.array(decode(matrix))

def round_to_int(x):
    """ Rounds the array, and returns it with entries as numpy integers """
    x = np.array(x)
    return np.round(x).astype(int)
