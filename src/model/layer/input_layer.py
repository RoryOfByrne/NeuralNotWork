from .layer import Layer

class InputLayer(Layer):
    def __init__(self, neuron_count, network):
        Layer.__init__(self, neuron_count, False)
        self.network = network

    def compute(self):
        if(self.inputs == None):
            self.inputs = self.network.input_vector
        return self.inputs