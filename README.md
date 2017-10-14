# Inquiry

Inquiry! A Numpy based neural network library which currently supports feed forward neural networks. Inquiry was initially created as an exploration of neural networks. 

Future plans include Cythonizing and adding support for recurrent layers.

Some examples of using Inquiry as presented in the examples section 

Creating neural networks is simple using Inquiry.

from Inquiry.network import *

n = NeuralNetwork()
n.add_input_layer(input_shape, layer_output_shape)
n.add_layer(layer_output_shape, output_shape)
n.add_output_layer() # You must add this to tell Inquiry that this is the final layer and "seal" the network

n.predict(inputs)

n.train(iterations, training_inputs, training_outputs) # This trains the neural network n using offline learning


