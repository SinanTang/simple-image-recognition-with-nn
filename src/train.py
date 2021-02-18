from src.prepare_data import prepare_training_data, scale_input_data_point
from src.simple_neural_network import NeuralNetwork
from src.utils import read_data_list


def train(input_list, target_list, epoch=5):
    input_nodes = 28 * 28
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    nn.train(input_list, target_list)

    return nn


if __name__ == "__main__":
    train_data = read_data_list('../data/mnist_train_100.csv')
    input_list, target_list = prepare_training_data(train_data)

    nn = train(input_list, target_list)

    test_data = read_data_list('../data/mnist_test_10.csv')[0].strip().split(',')[1:]
    test_datapoint = scale_input_data_point(test_data)
    output = nn.query(test_datapoint)
    label = output.argmax()
    print(label)
