from model import network
import numpy as np

training_data = np.array([[13, 3]])
training_labels = np.array([[0, 1, 0]])

n = network.Network(training_data,
                    training_labels, None, None)

n.addLayer(2, 'relu', is_input=True)
n.addLayer(5, 'relu')
n.addLayer(3, 'softmax', is_output=True)

n.train()
