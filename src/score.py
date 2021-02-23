import numpy

from src.prepare_data import prepare_training_data, scale_input_data_point
from src.train import train
from src.utils import read_data_list


def score():
    scorecard = []

    train_data = read_data_list('../data/mnist_train_100.csv')
    input_target_list = prepare_training_data(train_data)

    nn = train(input_target_list)

    test_data_list = read_data_list('../data/mnist_test_10.csv')

    for test in test_data_list:
        correct_label = int(test[0])
        test_input_raw = test.strip().split(',')[1:]
        test_datapoint = scale_input_data_point(test_input_raw)
        actual_output = nn.query(test_datapoint)
        actual_label = actual_output.argmax()

        if actual_label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size

    print('Accuracy score:', accuracy)
    print('Scorecard:', scorecard)


if __name__ == '__main__':
    score()
