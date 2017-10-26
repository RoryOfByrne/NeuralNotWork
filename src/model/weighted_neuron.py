from .neuron import Neuron

class WeightedNeuron(Neuron):
    def __init__(self, index):
        Neuron.__init__(self, index)
        self.input = None
        self.weights = None