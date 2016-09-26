import numpy
from math import e
class NeuralNet(object):

    def __init__(self, layer_node_counts, learning_rate):
        self._layer_counts = len(layer_node_counts)
        self._layer_node_counts = layer_node_counts
        self._learning_rate = learning_rate
        self._weights = []
        self._bias = []
        self._initialize_weights_and_bias()
        self._activation = numpy.vectorize(lambda v: 1/(1+pow(e, -v)))

    def _initialize_weights_and_bias(self):
        self._weights, self._bias = [], []
        for i, node_counts in enumerate(self._layer_node_counts):
            if i < self._layer_counts - 1:
                weight = numpy.random.normal(0.0,
                    pow(self._layer_node_counts[i+1],-0.5), (node_counts, self._layer_node_counts[i+1]))
                self._weights.append(weight)
                bias = numpy.random.normal(0.0,
                    pow(self._layer_node_counts[i+1],-0.5), self._layer_node_counts[i+1])
                self._bias.append(bias)

    def query(self,inputs):
        output_list = []
        input_list = []
        # get the outputs of input layer
        input_list.append(inputs)
        output_list.append(inputs)
        # get the outputs of hidden layers
        for i, weight in enumerate(self._weights):
            input_for_this_layer = output_list[-1].dot(weight) + self._bias[i]
            input_list.append(input_for_this_layer)
            output_list.append(self._activation(input_for_this_layer))
        return output_list

    def train(self, inputs, targets, max_iteration=10000, tolerance=0.01):
        targets_norm =  numpy.linalg.norm(targets)
        for i in range(max_iteration):
            # stop backpropagation if the outputs match the targets
            output_list = self.query(inputs)
            outputs = output_list[-1]
            errors = numpy.linalg.norm(outputs - targets)
            if errors/targets_norm < tolerance:
                break
            # get errors propagated back to each layer
            delta_list = [(outputs - targets) * outputs * (1-outputs)]
            ind = self._layer_counts - 2
            while ind >= 0:
                outputs = output_list[ind]
                current_delta = delta_list[0].dot(self._weights[ind].T) * outputs * (1-outputs)
                delta_list.insert(0, current_delta)
                ind -= 1
            # adjust the weights in each layer
            ind = self._layer_counts - 2
            while ind >= 0:
                outputs = output_list[ind]
                self._weights[ind] -=  outputs.T.dot(delta_list[ind+1]) * self._learning_rate
                self._bias[ind] -= sum(delta_list[ind+1]) * self._learning_rate
                ind -= 1

    
