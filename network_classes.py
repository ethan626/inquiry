#!/bin/python
import numpy as np
import itertools
from network_utilities import * 
np.random.seed(1)

class Layer():
    """ Hidden Layer of a Feed Forward Neural Network """

    def __init__(self, in_size, out_size, learning_rate=1, tanh=False):
        """ Constructor Method 
        
           This layer-> 
         this weight matrix                        Next Layer |
                                                              v
        

        """ 
        self.weights = 2 * np.random.random((in_size, out_size)) - 1 
        self.input_size = in_size
        self.output_size = out_size
        self.deltas = None 
        self.neuron_outputs = None 
        self.neuron_inputs = None
        self.learning_rate = learning_rate
        self.tanh = False

    def get_weights(self):
        return self.weights
    
    def activation(self, x):
        """ Sigmoid activation function """
        if self.tanh:
            return np.tanh(x)
        else:
            return 1/(1+np.exp(-x))

    def derivative(self, x):
        """ Sigmoid derivative """
        if self.tanh:
            return 1-np.tanh(x) ** 2
        return x * (1-x)
    
    def predict(self, x):
        """ Layers prediction for input *x* """
        self.neuron_inputs = np.array(x)
        self.neuron_outputs = self.activation(x)
        return np.dot(self.activation(x), self.weights) 

    def get_deltas(self, next_layer=None, prev_layer=None, **kwargs): 
        """ Calcuates the deltas for a layer for an entire epoch """ 
        self.deltas = np.dot(next_layer.deltas, self.weights.T) * self.derivative(self.neuron_outputs) 
        return self.deltas
        
    def adjust_weights(self,  prev_layer=None, next_layer=None, *args, **kwargs):
        """ Adjusts the weights between the layer and the previous layer """
        deltas = next_layer.deltas
        adjustments = self.learning_rate * np.dot(self.neuron_outputs.T, deltas)
        self.weights += adjustments

class OutputLayer(Layer):
    """ Output Layer """

    def __init__(self, learning_rate=1, tanh=False):
        """ """
        self.deltas = None 
        self.neuron_outputs = None 
        self.neuron_inputs = None 
        self.learning_rate = learning_rate
        self.weights = None     # unused
        self.tanh = tanh

    def predict(self, x):
        """ """
        self.neuron_inputs = np.array(x)
        self.neuron_outputs = self.activation(x)
        return self.neuron_outputs

    def get_deltas(self, error, *args, **kwargs):
        """ """
        derivs = self.derivative(self.neuron_outputs)
        self.deltas = error * derivs
        return self.deltas

    def adjust_weights(self, error, *args, **kwargs):
        """ """
        self.get_deltas(error)
    
class InputLayer(Layer):
    """ Input Layer of the network """

    def __init__(self, in_size, out_size, learning_rate=1):
        """  Init method. Weights are initialized to the identity matrix as the raw data is passed forward"""
        self.weights = 2*np.random.random((in_size, out_size)) - 1 
        self.input_size = in_size
        self.deltas = None 
        self.neuron_outputs = None 
        self.neuron_inputs = None 
        self.learning_rate = learning_rate 
        
    def predict(self, x):
        """ Input Layer predict. Passes the raw data forward  """
        self.neuron_inputs = np.array(x)
        self.neuron_outputs = np.array(x)
        return np.dot(x, self.weights)
    
    def adjust_weights(self, error, prev_layer=None, next_layer=None, **kwargs): 
        """ Adjusts the weights between the layer and the previous layer """
        deltas = next_layer.deltas
        adjustments = self.learning_rate * np.dot(self.neuron_inputs.T, deltas)
        self.weights += adjustments 
        
class NeuralNetwork():
    """ Feed Forward Neural Network """

    def __init__(self):
        """ self.layers = a list of all layers in the network. Currently the hidden layers are square matricies of the same size as the input shape """
        self.layers = []

    def add_input_layer(self, input_shape, output_shape, learning_rate=1):
        """ Add an input layer to the network """
        self.layers.append(InputLayer(input_shape, output_shape, learning_rate=learning_rate))

    def add_output_layer(self, learning_rate=1, tanh=False):
        """ Add an output neuron to the network """
        self.layers.append(OutputLayer(learning_rate=learning_rate, tanh=tanh))

    def add_layer(self, input_shape, output_shape, learning_rate=1, tanh=False):
        """ Add a layer to the network """
        self.layers.append(Layer(input_shape, output_shape, learning_rate=learning_rate, tanh=tanh))

    def weights(self):
        """ Returns a numpy array of the weights of the network, not counting the input layer with the weights of the layers closest to the input layers coming first """
        return np.array([layer.weights for layer in self.layers])

    def deltas(self):
        """ Returns a matrix of the deltas of each layer """
        return np.array([layer.deltas() for layer in self.layers])

    def neuron_outputs(self):
        """ Neuron outputs for each layer """
        return np.array([layers.neuron_outputs for layer in self.layers])

    def predict(self, x, one_hot=False):
        """ Prediction of input *x*. *one_hot* returns a one hot encoded prediction  """
        for layer in self.layers: 
            x = layer.predict(x)
        return x

    def adjust_weights(self, error, **kwargs):
        """ Adjusting of the Layer's weights """

        if len(self.layers[:-1]) > 2:  
            reversed_layers = [layer for layer in reversed(self.layers[:-1])]
            next_layer = self.layers[-1]
            next_layer.adjust_weights(error)
            
            for prev_layer, layer in zip(reversed_layers[1:], reversed_layers):
                layer.adjust_weights(error=error, prev_layer=prev_layer, next_layer=next_layer)
                layer.get_deltas(next_layer=next_layer, prev_layer=prev_layer)
                next_layer = layer

            self.layers[0].adjust_weights(error=error, next_layer=next_layer)
            
        else:
            self.layers[-2].adjust_weights(error=error)
            self.layers[-3].adjust_weights(error=error, next_layer=self.layers[1])

    def train(self, iters, training_set_inputs, training_set_outputs, error_vector=None, yield_error=None):
        """ Train the network for *iters* epochs. 
           *yield_error* turns this function into a generator which will yield the error values for each training iteration""" 
        for i in range(iters):
            prediction = self.predict(training_set_inputs) 
            error = training_set_outputs - prediction 
            self.adjust_weights(error)

            if yield_error:   # Sends the error to some generator
                yield error
