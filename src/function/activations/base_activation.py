class BaseActivation():
    def __init__(self):
        pass

    def element_wise(self, x):
        raise NotImplementedError

    def vector_wise(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError