from function.activation_fn import *
from .layer.input_layer import InputLayer
from .layer.weighted_layer import WeightedLayer
from util import *
from pprint import pprint


class Network():

    def __init__(self, train_d, train_l, test_d, test_l, learning_rate=0.5):
        '''
        Initialize stuff
        '''
        self.layers = []
        self.shape = []
        self.train_d = train_d
        self.train_l = train_l
        self.input_vector = None

    def addLayer(self, neuron_count, activation_fn, is_output=False):
        '''
        This will add a fully-connected layer.py to the network with `neuron_count` neurons

        :param neuron_count:
        :param bias:
        :return:
        '''
        layer = self.create_layer(neuron_count, activation_fn, is_output)
        self.layers.append(layer)
        self.shape.append(layer.neuron_count)

    def create_layer(self, neuron_count, activation_fn=None, is_output=False):
        is_input = (len(self.layers) == 0)

        hidden = (not is_input) and (not is_output)
        if(hidden):
            neuron_count += 1

        if(is_input):
            layer = InputLayer(neuron_count, self)
        else:
            if(hidden):
                weights = np.random.rand(neuron_count, self.shape[-1] + 1)
            else:
                weights = np.random.rand(neuron_count, self.shape[-1])

            print("Vector for Layer #%i" % (len(self.shape) + 1))
            showMatrix(weights, 3)
            layer = WeightedLayer(neuron_count, weights, activation_fn, self.layers[-1], hidden)

        return layer

    def clean_values(self):
        for layer in self.layers:
            layer.reset_inputs()

    def backprop(self, outputs):
        for i, l in enumerate(reversed(self.layers)):
            if(i == len(self.layers) - 1):
                erro

    def train(self):
        for sample, label in zip(self.train_d, self.train_l):
            self.input_vector = sample
            outputs = self.layers[-1].compute()

            print("\n---     OUTPUT     ---")
            pprint(outputs)

            self.backprop(outputs)

            self.clean_values()

            # Backprop
