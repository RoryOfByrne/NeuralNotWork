from model import network
import numpy as np
import random

with open('/home/rory/projects/neuralnotwork/data/winequality-red.csv') as f:
    raw = f.read().split('\n')

training_total = [r.split(';') for r in raw][1:]
training_data = [sample[:-1] for sample in training_total]
training_labels = [sample[-1] for sample in training_total]

training_data = [[float(i) for i in j] for j in training_data]
for i, j in enumerate(training_labels):
    if(j == ''):
        training_labels.pop(i)
        training_data.pop(i)
training_labels = [int(l) for l in training_labels]

def small_num():
    return random.uniform(0, 1)

training_labels = [n + small_num() for n in training_labels]

n_sig = network.Network(training_data,
                    training_labels, None, None)

n_sig.addLayer(11, 'sigmoid')
n_sig.addLayer(15, 'sigmoid')
n_sig.addLayer(4, 'sigmoid')
n_sig.addLayer(1, 'linear', is_output=True)


n_sig.train()
