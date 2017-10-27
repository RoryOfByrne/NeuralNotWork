import numpy as np
from function import activation_fn

class Neuron():
    def __init__(self):
        self.local_derivatives = []

    def reset_data(self):
        raise NotImplementedError

    def calc_local_derivatives(self):
        '''
        Here we calculate local derivatives which can be used during back-propagation
        :return:
        '''

        self.local_derivatives = [1, 2, 3]

    def fire(self):
        raise NotImplementedError



