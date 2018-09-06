from function.losses import mse as mse
from .layer.input_layer import InputLayer
from .layer.weighted_layer import WeightedLayer
import numpy as np


class Network():

    def __init__(self, train_d, train_l, test_d, test_l, learning_rate=0.5):
        '''
        Initialize stuff
        '''
        self.layers = []
        self.shape = []
        self.train_d = train_d
        self.train_l = train_l
        self.input_vector = None
        self.learning_rate = learning_rate

    def addLayer(self, neuron_count, activation_fn, is_output=False):
        '''
        This will add a fully-connected `Layer` to the network with `neuron_count` neurons

        :param neuron_count:
        :param bias:
        :return:
        '''
        layer = self._create_layer(neuron_count, activation_fn, is_output)
        self.layers.append(layer)
        self.shape.append(layer.neuron_count)

    def train(self):
        for e in range(10):
            print("EPOCH %s" % e)
            epoch_loss = 0.0
            sample_count = len(self.train_d)
            for sample, label in zip(self.train_d, self.train_l):
                self.input_vector = sample
                output = self.layers[-1].compute()

                # print("\nin: %s\nout: %s\nguess: %s" % (sample, label, output))

                loss_fn = mse.MSE()
                loss = loss_fn.element_wise(output, label)
                epoch_loss += loss
                # print("ERROR: %s" % loss)

                self._backprop(loss_fn, output, label)

                self._clean_values()
            print("Average loss: %s" % (epoch_loss/sample_count))
            print("\n")

    def _create_layer(self, neuron_count, activation_fn=None, is_output=False):
        is_input = (len(self.layers) == 0)


        if(is_input):
            layer = InputLayer(neuron_count, self)
        else:
            weights = np.random.rand(neuron_count, self.shape[-1])

            layer = WeightedLayer(neuron_count, weights, activation_fn, self.layers[-1])

        return layer

    def _add_bias(self, layer):
        pass

    def _clean_values(self):
        for layer in self.layers:
            layer.reset_inputs()

    def _backprop(self, loss_fn, output, label):
        prop_error = np.array(loss_fn.vector_wise_derivative(output, label))
        prop_error.reshape(prop_error.shape[0], 1)
        # 0 (first loop) is the last item
        for i, l in enumerate(reversed(self.layers)):
            if(i == len(self.layers)-1):
                continue
            if(i == 0): # Output Layer
                # np array(prop_error.size, prev_layer.real_inputs.size)
                l.incoming_deriv = prop_error

                l.local_derivatives()
                input_ders = l.inputs_derivs
                prop_error = input_ders
            else:
                l.incoming_deriv = prop_error
                l.local_derivatives()

                prop_error = l.inputs_derivs

        for i, l in enumerate(self.layers):
            if i == 0:
                continue
            l.update_weights(0.05)
