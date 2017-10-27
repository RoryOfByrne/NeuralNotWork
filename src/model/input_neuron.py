from .neuron import Neuron

class InputNeuron(Neuron):
    def __init__(self, index, network):
        Neuron.__init__(self)
        self.network = network
        self.index = index
        self.input = None

    def reset_input(self):
        self.input = None

    def fire(self):
        if not self.input:
            self.input = self.network.input_vector[self.index]
        return self.input
