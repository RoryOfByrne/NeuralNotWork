import function.activation_fn
import numpy as np
from .layer import Layer

class WeightedLayer(Layer):

    def __init__(self, neuron_count, weights, activation_fn_name, prev_layer, hidden):
        Layer.__init__(self, neuron_count, hidden)
        self.weights = weights
        self.prev_layer = prev_layer
        self.activation_fn = function.activation_fn.get_fn(activation_fn_name)


    def apply_activation_fn(self,signal):
        return self.activation_fn(signal)

    def compute(self):
        if(self.inputs == None):
            if(self.hidden):
                prev = self.prev_layer.compute()
                self.inputs = np.append(prev, self.bias)
            else:
                self.inputs = self.prev_layer.compute()

        signal = np.dot(self.weights, self.inputs)
        return self.apply_activation_fn(signal)