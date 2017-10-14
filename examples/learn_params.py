#!/bin/python
import sys
sys.path.append("../../")
from Inquiry.utilities import *
from Inquiry.network import *

""" 

What is the ideal number of hidden layers for a classification task using 1000 training iterations?

The hyper_param_network will figure it out!

The hyper_param_network is trained with online learning and the network n is trained with offline learning. 

Author: Ethan Henderson 
Github @ethan626

"""

""" Training Data """
ins = np.array(list(itertools.permutations([i for i in range(3)], 3)))
outs = np.array([[i % 2 for i in range(len(ins))]]).T

hyper_ins = 1

""" Creating the networks """
hidden_layer_network = NeuralNetwork() # This is the network which will learn the number of hidden layers for network n 
hidden_layer_network.add_input_layer(1, 10, learning_rate=.01) 
hidden_layer_network.add_layer(10, 1, learning_rate=.01)
hidden_layer_network.add_output_layer()

hidden_layers = np.int(hidden_layer_network.predict(hyper_ins)[0] * 10)

learning_rate_network = NeuralNetwork() # This is the network which will learn the learning rate for network n 
learning_rate_network.add_input_layer(1, 10, learning_rate=.01) 
learning_rate_network.add_layer(10, 1, learning_rate=.01)
learning_rate_network.add_output_layer()

neurons = 10 
learning_rate = learning_rate_network.predict(hyper_ins)
len_epoch = 500
iters = 1000

print('Training...')    

for j in range(len_epoch):
    n = NeuralNetwork() # Network which will classify vectors
    n.add_input_layer(3, neurons, learning_rate=learning_rate)

    for i in range(hidden_layers): # Number of hidden layers predicted by hidden_layer_network  
        n.add_layer(neurons, neurons, learning_rate=learning_rate)

    n.add_layer(neurons, 1)
    n.add_output_layer()

    n.train(iters, ins, outs)

    error_vector = n.predict(ins) - outs

    for error in error_vector:  # Adjust weights for each training input one at a time.  
        hidden_layer_network.adjust_weights(error) 

    hidden_layers = np.int(hidden_layer_network.predict(hyper_ins)[0] * 10)
