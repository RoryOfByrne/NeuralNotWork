from .neuron import Neuron

class WeightedNeuron(Neuron):
    def __init__(self, weights):
        Neuron.__init__(self)
        self.input = None
        self.weights = weights