from .weighted_neuron import WeightedNeuron
import numpy as np
class OutputNeuron(WeightedNeuron):
    def __init__(self, index):
        WeightedNeuron.__init__(self, index)
        self.input = np.array([])
        self.prev_neurons = []

    def reset_input(self):
        self.input = np.array([])

    def fire(self, next_index):
        if (self.input.size == 0):
            synaptic_input = []
            for n in self.prev_neurons:
                synaptic_input.append(n.fire(self.index))
            self.input = np.array(synaptic_input)

        # We use the rows of the weight matrix, not cols
        local_weights = self.weights[next_index]
        signal = (np.dot(local_weights, self.input))
        return signal
