import function.activation_fn
import numpy as np

class Layer():

    def __init__(self, neuron_count):
        self.inputs = None
        self.neuron_count = neuron_count

    def reset_inputs(self):
        self.inputs = None

    def compute(self):
        raise NotImplementedError