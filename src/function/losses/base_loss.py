class BaseLoss():
    def __init__(self):
        pass

    def element_wise(self, x, actual):
        raise NotImplementedError

    def vector_wise(self, x, actual):
        raise NotImplementedError

    def derivative(self, x, actual):
        raise NotImplementedError