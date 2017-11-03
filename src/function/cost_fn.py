import numpy as np

def cross_entropy(estimate, label):
    estimate_logs = np.log(estimate)

    return -(np.sum(
        np.multiply(label, estimate_logs)
        +
        np.multiply(1-label, estimate_logs)))