import numpy


def scale_input_data_point(input_data):
    return numpy.asfarray(input_data) / 255.0 * 0.99 + 0.01


def prepare_training_data(data_list):
    input_list, target_list = [], []

    for record in data_list:
        all_values = record.strip().split(',')
        # scale the input values
        input = scale_input_data_point(all_values[1:])
        input_list.append(input)
        # construct the target matrix
        target = numpy.zeros(10) + 0.01
        target[int(all_values[0])] = 0.99
        target_list.append(target)

    return input_list, target_list


