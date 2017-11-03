import function.activation_fn
import numpy as np
from .layer import Layer

class WeightedLayer(Layer):

    def __init__(self, neuron_count, weights, activation_fn_name, prev_layer):
        Layer.__init__(self, neuron_count)
        self.weights = weights
        self.prev_layer = prev_layer
        self.activation_fn = function.activation_fn.get_fn(activation_fn_name)

        self.outputs = None
        self.incoming_deriv = None
        self.weights_derivs = None
        self.inputs_derivs = None

    def local_derivatives(self):
        #dE/dW
        self.weights_derivs = np.outer(self.incoming_deriv, self.inputs)

        self.inputs_derivs = np.dot(self.weights.T, self.incoming_deriv)

    def update_weights(self, learning_rate):
        self.weights = self.weights - np.multiply(learning_rate, self.weights_derivs)

    def reset_inputs(self):
        self.inputs = None

    def apply_activation_fn(self):
        return self.activation_fn.vector_wise(self.signal)

    def compute(self):
        if(self.inputs == None):

            self.inputs = self.prev_layer.compute()
            self.inputs = self.inputs.reshape(self.inputs.shape[0], -1)

        self.signal = np.dot(self.weights, self.inputs)

        self.outputs = self.apply_activation_fn()

        return self.outputs