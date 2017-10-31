from model import network
import numpy as np

training_data = np.array([[13, 3]])
training_labels = np.array([[0, 1, 0]])

n_sig = network.Network(training_data,
                    training_labels, None, None)

n_sig.addLayer(2, 'sigmoid_vec')
n_sig.addLayer(5, 'sigmoid_vec')
n_sig.addLayer(3, 'softmax', is_output=True)


n_sig.train()
