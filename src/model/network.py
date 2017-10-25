import numpy as np
from .neuron import Neuron
from function.activation_fn import *

class Network():

    def __init__(self):
        '''
        Initialize stuff
        '''
        self.layers = []
        self.network_shape = []

    def addLayer(self, neuron_count, bias):
        '''
        This will add a fully-connected layer.py to the network with `neuron_count` neurons

        :param neuron_count:
        :param bias:
        :return:
        '''

        self.layers.append(self.create_layer(neuron_count, 5, bias, activation_fn=relu))

    def create_layer(self, neuron_count, input_dim, bias, activation_fn):
        self.network_shape.append(neuron_count)
        neurons = []
        weights = np.random.rand(input_dim, neuron_count)
        is_input_layer = len(self.layers) == 0

        for _ in neuron_count:
            neuron = Neuron(activation_fn, is_input_layer, bias)
            if(len(self.layers) == 0):
                neuron.is_input_neuron = True
            else:
                neuron.is_input_neuron = False
                prev_layer = self.layers[-1]
                for n in prev_layer[1]:
                    neuron.prev_neurons.append(n)
                    n.next_neurons.append(self)

            neurons.append(neuron)

        return (weights, neurons)

    def run(self):

        return run_rec(len(self.layers)-1)

    def run_rec(self, layer_num):
        if layer_num == 0:
            return foo
        elif layer_num == len(self.layers):
            

        layer = self.layers[layer_num]

        return run_rec(layer_num-1)


class Layer()
    def __init__(self, weights):
        self.weights = weights
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def run_layer(self):
        output = []
        for n in self.neurons:
