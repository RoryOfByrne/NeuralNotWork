from .weighted_neuron import WeightedNeuron

class BiasNeuron(WeightedNeuron):
    def __init__(self, index, bias=1):
        WeightedNeuron.__init__(self, index)
        self.input = bias

    def reset_input(self):
        pass # Bias never changes

    def fire(self, next_index):
        return self.input
