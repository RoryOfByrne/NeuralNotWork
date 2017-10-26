import numpy as np
from .neuron import Neuron
from .bias_neuron import BiasNeuron
from .hidden_neuron import HiddenNeuron
from .input_neuron import InputNeuron
from .output_neuron import OutputNeuron
from .weighted_neuron import WeightedNeuron
from function.activation_fn import *

class Network():

    def __init__(self, train_d, train_l, test_d, test_l):
        '''
        Initialize stuff
        '''
        self.layers = []
        self.shape = []
        self.train_d = train_d
        self.train_l = train_l
        self.input_vector = None

    def addLayer(self, neuron_count, activation_fn, is_output=False, is_input=False):
        '''
        This will add a fully-connected layer.py to the network with `neuron_count` neurons

        :param neuron_count:
        :param bias:
        :return:
        '''
        self.layers.append(self.create_layer(neuron_count, activation_fn, is_output, is_input))

    def create_layer(self, neuron_count, activation_fn, is_output, is_input):
        self.shape.append(neuron_count)
        neurons = []

        if not is_output and not is_input:
            bias_neuron = BiasNeuron(self, 1)
            neurons.append(bias_neuron)

        if(activation_fn == 'relu'):
            act_fn = relu
        elif(activation_fn == 'softmax'):
            act_fn = None

        for neuron_index in range(neuron_count):
            if is_input:
                neuron = InputNeuron(neuron_index, self)
            elif is_output:
                neuron = OutputNeuron(neuron_index)
                prev_layer = self.layers[-1]
                for n in prev_layer:
                    neuron.prev_neurons.append(n)
            else:
                neuron = HiddenNeuron(activation_fn,
                                neuron_index)
                prev_layer = self.layers[-1]
                for n in prev_layer:
                    neuron.prev_neurons.append(n)

            neurons.append(neuron)

        return neurons

    def set_up_weights(self):
        for i, layer in enumerate(self.layers):
            if i == 0: # Input layer - no weights
                continue

            for n in layer:
                if i == len(self.layers) - 1: # Output layer
                    weights = np.random.rand(1, len(self.layers[i - 1]))
                else: # Hidden layer
                    weights = np.random.rand(len(self.layers[i+1]), len(self.layers[i-1]))

                n.weights = weights

    def clean_values(self):
        for i, layer in enumerate(self.layers):
            for neuron in layer:
                neuron.reset_input()

    def train(self):
        self.set_up_weights()
        for sample, label in zip(self.train_d, self.train_l):
            outputs = []
            self.input_vector = sample
            for i, n in enumerate(self.layers[-1]):
                outputs.append(n.fire(0))

            print(outputs)
            softmax_result = softmax(np.array(outputs))
            print(softmax_result)
            print("\n")

            self.clean_values()

            # Backprop
