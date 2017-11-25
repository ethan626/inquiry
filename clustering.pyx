import numpy as np
cimport numpy as np

cpdef double norm(double[:] x, double[:] y):
    """ Euclidean Distance Between vectors x and y """
    cdef unsigned int i
    cdef unsigned int vector_length = x.shape[0]
    cdef double total
    
    for i in range(vector_length):
        total += pow(x[i] - y[i], 2)
    
    return np.sqrt(total)

cpdef double[:,:] initial_points(double[:,:] data, unsigned short num_means):
    """ Assumes data is a numpy array. Sets the initial guesses for the clusters """
    cdef unsigned int i,j
    cdef unsigned int dim = data.shape[1]
    cdef double maxs, mins
    cdef double[:,:] initial_points = np.empty((num_means, dim))

    maxs = np.amax(data) 
    mins = np.amin(data)
    
    for i in range(num_means):
        for j in range(dim):
         initial_points[i,j] = np.random.random(1) * np.random.choice(np.array([maxs, mins])) # need a way to store two points, build a c struct?

    return initial_points
    
cpdef double [:,:] mean(double[:, :] data):
    """ Data must be symetric vectors """

    cdef unsigned int i, j, k
    cdef unsigned int num_points = data.shape[0]
    cdef unsigned int dimensions = data.shape[1]
    cdef double[:,:] mean_array = np.empty((1, dimensions))   

    if dimensions > 1:                    # Sum all first entries, then second etc 
        for k in range(num_points):
            j = k                         #Need to remember as which 'k' as it will be referenced out of the block 
            total = 0 
            for i in range(dimensions):
                total += data[j, i]

            mean_array[0, j] = total / num_points

    else:
        total = 0
        for i in range(num_points):
            total += data[i]            

        mean_array[0, j] = total / num_points
        
    return mean_array

cpdef unsigned char[:,:] classify(double[:,:] data, double[:,:] means, unsigned short num_means):
    cdef unsigned short i, j, class_index
    cdef unsigned short len_data = data.shape[0]
    cdef unsigned char [:,:] types = np.empty((1, len_data), dtype='uint8')
    
    cdef double[:] closest_mean = np.empty(len_data) 

    for i in range(len_data):        
        closest_mean, class_index = means[0], 0
        
        for j in range(num_means):
            if norm(data[i], means[j]) < norm(data[i], closest_mean):
                closest_mean, class_index = means[j], j

        types[0, i] = class_index

    return types
