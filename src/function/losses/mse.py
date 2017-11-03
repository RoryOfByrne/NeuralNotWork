from .base_loss import BaseLoss
import numpy as np

class MSE(BaseLoss):
    def __init__(self):
        BaseLoss.__init__(self)

    def element_wise(self, x, actual):
        return 1/2 * np.square((x - actual))

    def vector_wise(self, x, actual):
        vfunc = np.vectorize(lambda o, a: self.element_wise(o, a))
        return vfunc(x, actual)

    def vector_wise_derivative(self, x, actual):
        vfunc = np.vectorize(lambda o, a: self.derivative(o, a))
        return vfunc(x, actual)

    def derivative(self, x, actual):
        return x - actual