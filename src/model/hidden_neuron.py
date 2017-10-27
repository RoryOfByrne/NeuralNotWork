import numpy as np
from function import activation_fn
from .weighted_neuron import WeightedNeuron

class HiddenNeuron(WeightedNeuron):
    def __init__(self, activation_fn_name, weights):
        WeightedNeuron.__init__(self, weights)

        self.activation_fn = activation_fn.get_fn(activation_fn_name)
        self.prev_neurons = []

        self.input = np.array([])

    def calc_local_derivatives(self):
        '''
        Here we calculate local derivatives which can be used during back-propagation
        :return:
        '''

        self.local_derivatives = [1, 2, 3]

    def apply_activation_fn(self, input):
        '''
        Apply the activation function to `self.combined_value` and return the answer
        :return:
        '''

        return self.activation_fn(input)

    def reset_input(self):
        self.input = np.array([])

    def fire(self):
        if (self.input.size == 0):
            synaptic_input = []
            for n in self.prev_neurons:
                synaptic_input.append(n.fire())
            self.input = np.array(synaptic_input)

        # We use the rows of the weight matrix, not cols
        signal = np.dot(self.weights, self.input)
        return self.apply_activation_fn(signal)



