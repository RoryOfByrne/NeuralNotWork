from .layer import Layer
import numpy as np

class InputLayer(Layer):
    def __init__(self, neuron_count, network):
        Layer.__init__(self, neuron_count)
        self.network = network

    def compute(self):
        if(self.inputs == None):
            self.inputs = np.array(self.network.input_vector)
            self.inputs = self.inputs.reshape(self.inputs.shape[0], -1)
        return self.inputs