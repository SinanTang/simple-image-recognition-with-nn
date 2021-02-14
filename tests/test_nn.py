import unittest

from src.simple_neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def test_nn_init_query(self):
        input_nodes = 3
        hidden_nodes = 3
        output_nodes = 3

        learned_rate = 0.1

        n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learned_rate)

        outputs = n.query([1.0, 0.5, -1.5])

        self.assertEqual(3, len(outputs))
        self.assertEqual(1, len(outputs[0]))
