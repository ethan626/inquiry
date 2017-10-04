#!/bin/python
from network_utilities import *
from network_classes import *
from network_wrappers import *  # NIU 
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
np.random.seed(1)
""" 
Example network which preforms a classification task. 

Feed Forward Neural Network with backpropogation 

This project classifies vectors as being one of two types, type 1 or type 0. 

The current setting utilizes a network with an input layer, two hidden layers, and an output layer. 

Network topology is adjusted by calling the various add_layer functions of the NeuralNetwork class. 

This network utilizes offline learning, all training data is presented to the network in a matrix. Weight adjustments are done at the end of the epoch, which is represented by the matrix of training data. 
    
Visualization of the classification results after training is done with Matplotlib and is displayed by default.  

Author: Ethan Henderson 
Github: ethan626

"""

# if __name__ == '__main__':       
types = 3 
vector_size = 3
ins = np.array(list(itertools.permutations([i for i in range(types)], vector_size)))
outs = one_hot_encode(np.array([[i % types for i in range(len(ins))]]).T)
learning_rate = .01

n = NeuralNetwork()

n.add_input_layer(vector_size, 40, learning_rate=learning_rate)
n.add_layer(40, 40, learning_rate=learning_rate)
n.add_layer(40, types, learning_rate=learning_rate)
n.add_output_layer(learning_rate=learning_rate)

iters = 10000

print('Training...')

n.train(iters, ins, outs)

results = n.predict(ins)
results = round_to_int(results)
one_hot_results = one_hot_decode(results)
        
""" 
Here we plot the sum magnitude of the error during training.
"""

# plt.figure()
# ax = plt.gca()
# ax.set_facecolor('black')
# plt.title("Error in Training")
# plt.xlabel('Training Epochs')
# plt.ylabel('Error')
# plt.plot([i for i,j in enumerate(error_list)], error_list, color='cyan')
# plt.show()
# plt.close()

"""

isualization of the classification 

ere we plot the points which the network classifies after training as  type 1 in red and of type 0 in blue. 

# """
# zeros = ins[::3]
# ones = ins[1::3]
# twos = ins[2::3]

# figure = plt.figure()

# ax = figure.gca(projection='3d')
# ax.axis('off')
# ax.set_facecolor('black')
# ax.patch.set_facecolor('black')
# ax.set_title('Classication of Vectors', color='w')

# x,y,z = np.split(zeros, 3, 1) # Split vectors into components so that we can plot. Variables x,y,z will be reused for plotting.
# ax.scatter(x,y,z,color='red', label='Training Data Zero', s=300)

# x,y,z = np.split(ones, 3, 1)   
# ax.scatter(x,y,z,color='grey', label='Training Data One', s=300)


# x,y,z = np.split(twos, 3, 1)  
# ax.scatter(x,y,z,color='green', label='Training Data Two', s=300)

# randoms = np.array([np.random.random(3) for i in range(1000)]) * 5
# results  = np.array([np.round(n.predict(i)) for i in randoms])

# results = round_to_int(results)
# results = one_hot_decode(results)

# new_zeros = np.array([i for i,j in zip(randoms, results) if j == 0])
# new_ones = np.array([i for i,j in zip(randoms, results) if j == 1])
# new_twos = np.array([i for i,j in zip(randoms, results) if j == 2])

# # These are placed in try/except in the case that new_ones/new_zeroes is an empty list, causing an error.
# try:
#     x,y,z = np.split(new_ones, 3, 1)  
#     ax.scatter(x,y,z, 'grey', label='Predicted as One')

# except Exception as e:
#     print(e)

# try:
#     x,y,z = np.split(new_zeros, 3, 1)
#     ax.scatter(x,y,z, 'red', label='Predicted as Zero')

# except Exception as e:
#     print(e)

# try:
#     x,y,z = np.split(new_twos, 3, 1)
#     ax.scatter(x,y,z, 'green', label='Predicted as Two')

# except Exception as e:
#     print(e)

# ax.set_xlabel('x', color='w')
# ax.set_ylabel('y', color='w')
# ax.set_zlabel('z', color='w')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.show()
# plt.close()
