import numpy as np

def relu(x):
    if x > 0:
        return x
    return 0

def relu_vec(x):
    vfunc = np.vectorize(lambda o: max(0, o))
    return vfunc(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    #return np.exp(x_vector) / float(sum(np.exp(x_vector)))

def get_fn(name):
    if(name == 'relu'):
        return relu
    elif(name == 'softmax'):
        return softmax
    elif(name == 'relu_vec'):
        return relu_vec