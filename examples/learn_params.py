#!/bin/python
from network_utilities import *
from network_classes import * 

""" 

Learning the hyperparameters.

Currently the hyperparameters are the learning rate and the number of neurons. 

Author: Ethan Henderson 
Github @ethan626

"""

ins = np.array(list(itertools.permutations([i for i in range(3)], 3)))
outs = np.array([[i % 2 for i in range(len(ins))]]).T
len_epoch = 10
epochs = 10 
""" Creating the networks """
hyper_param_network = NeuralNetwork() # This is the network which will learn the hyperparameters for network n 
hyper_param_network.add_input_layer(1, 10)             #Takes  input the error
hyper_param_network.add_layer(10, 1)
hyper_param_network.add_output_layer(learning_rate=.1)

a = hyper_param_network         # For ease in terminal  

hidden_layers = np.int(hyper_param_network.predict(1)[0] * 10)
neurons = 10 
learning_rate = .01
iters = 200 
error_vector = np.array([])
    
for j in range(len_epoch):
    """ Here we generate the new network """
    n = NeuralNetwork() # Network which will classify vectors
    n.add_input_layer(3, neurons, learning_rate=learning_rate)

    for i in range(hidden_layers):
        n.add_layer(neurons, neurons, learning_rate=learning_rate)

    n.add_layer(neurons, 1)
    n.add_output_layer()

    n.train(iters, ins, outs)

    error = np.sum(abs(n.predict(ins) - outs)) # Get error after training

    hyper_param_network.adjust_weights(np.array([error]))
    hidden_layers = np.int(hyper_param_network.predict(1)[0] * 10)
