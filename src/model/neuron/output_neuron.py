from .weighted_neuron import WeightedNeuron
import numpy as np
class OutputNeuron(WeightedNeuron):
    def __init__(self, weights):
        WeightedNeuron.__init__(self, weights)
        self.input = np.array([])
        self.prev_neurons = []

    def reset_input(self):
        self.input = np.array([])

    def fire(self):
        if (self.input.size == 0):
            synaptic_input = []
            for n in self.prev_neurons:
                synaptic_input.append(n.fire())
            self.input = np.array(synaptic_input)

        signal = (np.dot(self.weights, self.input))
        return signal
