import function.activation_fn
import numpy as np

class Layer():

    def __init__(self, neuron_count, hidden, bias=1):
        self.inputs = None
        self.neuron_count = neuron_count
        self.bias = bias
        self.hidden = hidden

    def reset_inputs(self):
        self.inputs = None

    def compute(self):
        raise NotImplementedError