import numpy
import matplotlib.pyplot as plt

from src.utils import read_data_list


def visualise_datapoint(data_list, n):
    all_values = data_list[n].split(',')[1:]

    image_array = numpy.asfarray(all_values).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


if __name__ == '__main__':
    data_list = read_data_list('../../data/mnist_test_10.csv')
    visualise_datapoint(data_list, 9)
