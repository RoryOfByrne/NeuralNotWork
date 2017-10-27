from .weighted_neuron import WeightedNeuron

class BiasNeuron(WeightedNeuron):
    def __init__(self, weights, bias=1):
        WeightedNeuron.__init__(self, weights)
        self.input = bias

    def reset_input(self):
        pass # Bias never changes

    def fire(self):
        return self.input
