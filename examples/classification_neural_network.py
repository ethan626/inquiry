#!/bin/python
import sys
sys.path.append("../../")
from Inquiry.utilities import *
from Inquiry.network import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
np.random.seed(1)

""" 
Example network which preforms a classification task. 

Feed Forward Neural Network with backpropogation 

This project classifies vectors as being one of three types, type 0, type 1, or type 2. 

The current setting utilizes a network with an input layer, two hidden layers, and an output layer. 

Network topology is adjusted by calling the various add_layer functions of the NeuralNetwork class. 

This network utilizes offline learning, all training data is presented to the network in a matrix. Weight adjustments are done at the end of the epoch, which is represented by the matrix of training data. 
    
Visualization of the classification results after training is done with Matplotlib and is displayed by default.  

Author: Ethan Henderson 
Github: ethan626

"""

if __name__ == '__main__':       
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

    """

    isualization of the classification 

    ere we plot the points which the network classifies after training as  type 1 in red and of type 0 in blue. 

    """
    zeros = ins[::3]
    ones = ins[1::3]
    twos = ins[2::3]

    figure = plt.figure()

    ax = figure.gca(projection='3d')
    ax.axis('off')
    ax.set_facecolor('black')
    ax.patch.set_facecolor('black')
    ax.set_title('Classication of Vectors', color='w')

    x,y,z = np.split(zeros, 3, 1) # Split vectors into components so that we can plot. Variables x,y,z will be reused for plotting.
    ax.scatter(x,y,z,color='red', label='Training Data Zero', s=300)

    x,y,z = np.split(ones, 3, 1)    # Now we plot the "ones" 
    ax.scatter(x,y,z,color='grey', label='Training Data One', s=300)

    x,y,z = np.split(twos, 3, 1)  
    ax.scatter(x,y,z,color='green', label='Training Data Two', s=300)

    randoms = np.array([np.random.random(3) for i in range(1000)]) * 5 # Generate some random points  
    random_predictions  = np.array([np.round(n.predict(i)) for i in randoms])

    random_predictions = round_to_int(random_predictions)
    random_predictions = one_hot_decode(random_predictions)

    new_zeros = np.array([i for i,j in zip(randoms, random_predictions) if j == 0])
    new_ones = np.array([i for i,j in zip(randoms, random_predictions) if j == 1])
    new_twos = np.array([i for i,j in zip(randoms, random_predictions) if j == 2])

    # These are placed in try/except in the case that new_ones/new_zeroes is an empty list, causing an error.
    try:
        x,y,z = np.split(new_ones, 3, 1)  
        ax.scatter(x,y,z, 'grey', label='Predicted as One')

    except Exception as e:
        print(e)

    try:
        x,y,z = np.split(new_zeros, 3, 1)
        ax.scatter(x,y,z, 'red', label='Predicted as Zero')

    except Exception as e:
        print(e)

    try:
        x,y,z = np.split(new_twos, 3, 1)
        ax.scatter(x,y,z, 'green', label='Predicted as Two')

    except Exception as e:
        print(e)

    ax.set_xlabel('x', color='w')
    ax.set_ylabel('y', color='w')
    ax.set_zlabel('z', color='w')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()
    plt.close()
