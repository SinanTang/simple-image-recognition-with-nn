import numpy


def scale_input_data_point(input_data):
    return numpy.asfarray(input_data) / 255.0 * 0.99 + 0.01


def prepare_training_data(data_list):
    """
    Preprocess training dataset so it can be used for training.
    :param data_list:
    :return: [(intput1, target1), (input2, target2), ...]
    """
    input_target_list = []

    for record in data_list:
        all_values = record.strip().split(',')
        # scale the input values
        input = scale_input_data_point(all_values[1:])
        # construct the target matrix
        target = numpy.zeros(10) + 0.01
        target[int(all_values[0])] = 0.99

        input_target_list.append((input, target))

    return input_target_list


