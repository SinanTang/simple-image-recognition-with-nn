import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # link weight matrices, Wih & Who
        self.wih = numpy.random.normal(loc=0.0,
                                       scale=pow(self.inodes, -0.5),
                                       size=(self.hnodes, self.inodes))
        self.who = numpy.random.normal(loc=0.0,
                                       scale=pow(self.hnodes, -0.5),
                                       size=(self.onodes, self.hnodes))

        # defaults to the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # feed forward the input signals through the 3-layer NN
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # back propagate errors
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        hidden_outputs.T)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        inputs.T)

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
