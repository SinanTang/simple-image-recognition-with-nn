def read_data_list(data_file):
    fh = open(data_file, 'r')
    data_list = fh.readlines()
    fh.close()

    return data_list
