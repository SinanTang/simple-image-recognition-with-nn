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
        # defaults to the logit function
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

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

    def backquery(self, target):
        """
        Back query the neural network to see the image produced by trained NN.
        :param target:
        :return:
        """
        # transpose the target list into a vertical array
        final_output = numpy.array(target, ndmin=2).T

        # calculate the signal into the output layer
        final_input = self.inverse_activation_function(final_output)

        # calculate the signal out of the hidden layer
        hidden_output = numpy.dot(self.who.T, final_input)
        # scale back to .01-.99
        hidden_output -= numpy.min(hidden_output)
        hidden_output /= numpy.max(hidden_output)
        hidden_output *= 0.98
        hidden_output += 0.01

        hidden_input = self.inverse_activation_function(hidden_output)

        inp = numpy.dot(self.wih.T, hidden_input)
        # scale the input back to .01-.99
        inp -= numpy.min(inp)
        inp /= numpy.max(inp)
        inp *= 0.98
        inp += 0.01

        return inp
