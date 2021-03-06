import numpy as np
from .activations.sigmoid import Sigmoid
from .activations.linear import Linear

### TODO: Make class-based implementations of these activation functions

def relu(x):
    if x > 0:
        return x
    return 0

def relu_vec(x):
    vfunc = np.vectorize(lambda o: relu(o))
    return vfunc(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    #return np.exp(x_vector) / float(sum(np.exp(x_vector)))

#####################################3

def get_fn(name):
    if(name == 'relu'):
        return relu
    elif(name == 'softmax'):
        return softmax
    elif(name == 'relu_vec'):
        return relu_vec
    elif(name == 'sigmoid'):
        act_fn = Sigmoid
        return act_fn()
    elif(name == 'linear'):
        act_fn = Linear
        return act_fn()