from src.prepare_data import prepare_training_data, scale_input_data_point
from src.simple_neural_network import NeuralNetwork
from src.utils import read_data_list


def train(input_target_list, hidden_nodes=200, epoch=7, learning_rate=0.1):
    """
    Train a NeuralNetwork instance based on the training data provided.
    :param hidden_nodes:
    :param input_target_list: [(intput1, target1), (input2, target2), ...]
    :param epoch: the number of times the training dataset is used for training
    :param learning_rate: High learning rate might lead to overshooting.
    :return: a trained NeuralNetwork instance
    """
    input_nodes = 28 * 28  # the pixels making up the image datapoint
    output_nodes = 10

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    for i in range(epoch):
        print('Training Epoch {}...'.format(i + 1))
        for (inp, target) in input_target_list:
            nn.train(inp, target)

    return nn


if __name__ == "__main__":
    train_data = read_data_list('../data/mnist_train_100.csv')
    input_target_list = prepare_training_data(train_data)

    nn = train(input_target_list)

    test_data = read_data_list('../data/mnist_test_10.csv')[1].strip().split(',')[1:]
    test_datapoint = scale_input_data_point(test_data)
    output = nn.query(test_datapoint)
    print(output)

    label = output.argmax()
    print(label)
