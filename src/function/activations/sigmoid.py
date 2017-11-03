from .base_activation import BaseActivation
import numpy as np

class Sigmoid(BaseActivation):
    def __init__(self):
        BaseActivation.__init__(self)

    def element_wise(self, x):
        return 1 / (1 + np.exp(-x))

    def vector_wise(self, x):
        vfunc = np.vectorize(lambda o: self.element_wise(o))
        return vfunc(x)

    def vector_wise_derivative(self, x):
        vfunc = np.vectorize(lambda o: self.derivative(o))
        return vfunc(x)

    def derivative(self, x):
        return x * (1.0 - x)