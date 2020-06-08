import joblib
import numpy as np
from keras.datasets import mnist
from rotated_cell_data_generator import DataGenerator



def get_data(params, data_generator=None):

    ret = {}

    # get data if not provided
    if data_generator is not None:
        x_train, x_test, y_train, y_test = load_data(params, data_generator=data_generator)
    else:
        print("WARNING: Using data provided in arguments. Must be tuple or array of format (x_train, x_test, y_train, y_test)")
        x_train, x_test, y_train, y_test = load_data(params)

    ret['spectral'] = {}


    x_val=x_test
    y_val=y_test


    ret['spectral']['train_and_test'] = (x_train, y_train, x_val, y_val, x_test, y_test)


    return ret


def load_data(params, data_generator=None):
    '''
    Convenience function: reads from disk, downloads, or generates the data specified in params
    '''
    if params['dset'] == 'mnist':
        x_train, x_test, y_train, y_test = get_mnist()
    elif params['dset'] == 'pfc':
        assert data_generator is not None
        x_train, x_test, y_train, y_test = get_pfc(data_generator)

    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return x_train, x_test, y_train, y_test


def get_pfc(data_generator, test_split=1):
    x_train = []
    y_train = []
    n_train_cells = int(data_generator.n_cells * test_split)
    for cell_i in range(n_train_cells):
        #print(cell_i)
        cell, label = data_generator.__getitem__(cell_i)
        x_train.append(cell[0])
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if test_split == 1:
        x_test = x_train
        y_test = y_train
    else:
        x_test = []
        y_test = []
        for cell_i in range(n_train_cells, data_generator.n_cells):
            #print(cell_i)
            cell, label = data_generator.__getitem__(cell_i)
            x_test.append(cell[0])
            y_test.append(label)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

    try:
        with open('sparse_matrices.sav', 'wb') as f:
            joblib.dump(data_generator.sparse_matrices, f)  # and save the sparse matrix dict for use later
    except MemoryError:
        print('Not enough memory to save')

    return x_train, x_test, y_train, y_test


def get_mnist():
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    return x_train, x_test, y_train, y_test