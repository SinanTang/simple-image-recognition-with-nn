import numpy
import matplotlib.pyplot as plt

from src.prepare_data import prepare_training_data
from src.train import train
from src.utils import read_data_list

if __name__ == '__main__':
    # Test backquery with label 0
    label = 9
    target = numpy.zeros(10) + 0.01
    target[label] = 0.99

    train_data = read_data_list('../data/mnist_train_100.csv')
    input_target_list = prepare_training_data(train_data)
    nn = train(input_target_list, hidden_nodes=100, epoch=5, learning_rate=0.1)

    image_data = nn.backquery(target)
    plt.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    plt.show()
