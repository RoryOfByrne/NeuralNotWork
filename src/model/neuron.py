

class Neuron():
    def __init__(self, activation_fn, is_input_neuron, bias):
        self.activation_fn = activation_fn
        self.prev_neurons = []
        self.next_neurons = []
        self.is_input_neuron = is_input_neuron

        self.local_derivatives = []
        self.combined_value = None
        self.bias = bias

    def

    def calc_local_derivatives(self):
        '''
        Here we calculate local derivatives which can be used during back-propagation
        :return:
        '''

    def apply_activation_fn(self):
        '''
        Apply the activation function to `self.combined_value` and return the answer
        :return:
        '''

        return self.activation_fn(self.combined_value)

    def fire(self, weight):
        return weight

