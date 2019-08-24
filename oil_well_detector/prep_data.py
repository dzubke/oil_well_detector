import numpy as np
from typing import List, Tuple
import numpy as np
import math
import rasterio as rio


def reshape(rst_fn: str) -> Tuple[np.ndarray, dict]:
    '''Takes in a raster image and reshapes it into a row vector

    Parameters
    ----------
    rst_fn -- a rasterio object whose values will be reshaped into a row vector
    
    Returns
    ----------
    rst_vec -- a column vector of the the numpy array associated with the 'img' rasterio object
    metadata -- the metadata associated with the 'img' rasterio object

    '''
    with rio.open(rst_fn, 'r') as rst:
        metadata = rst.meta.copy()
        array = rst.read(1)
        rst_vec = array.reshape(-1,1).T
        

    return rst_vec, metadata

def normalize(X_data: np.ndarray) -> np.ndarray:
    """Normalizes the values of the raster image

    Parameters
    ----------
    X_data: np.ndarray of shape number of bands (n) x number of pixels (m)
        A numpy array of the unrolled raster images

    Returns
    ----------
    X_data_norm: np.ndarray of shape number of bands (n) x number of pixels (m)
        A numpy array of the normalized unroled raster images

    """

    X_data_norm = X_data
   
    for i in range(X_data.shape[0]):
        #divides the whole array by the maximum value
        X_data_norm = X_data[i] / max(X_data[i])
    
    assert (X_data_norm.shape == X_data.shape), "The input and ouput should have the same shape"

    return X_data_norm


def data_split(X_data: np.ndarray, Y_data: np.ndarray, set_percent: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    """Splits the input x-array and labelled y-vector into training, developmment, and test sets.

    Parameters
    ----------
    X_data:  np.ndarray of shape number of bands (n) x number of pixels (m) 
        A numpy array of the unrolled raster images where n is the number of bands
    
    Y_data: np.ndarray of shape 1 x number of pixels (m)
        A column vector of the unrolled raster labels 

    set_percent: a list of decimals of length 3
        A list of the percentages of the total data to be allocated to the training, development, and test sets


    Returns
    ----------
    X_train: np.ndarray of the input training set
    X_dev: np.ndarray of the input development set
    X_test: np.ndarray of the input test set
    Y_train: np.ndarray of the label training set
    Y_dev: np.ndarray of the label development set
    Y_test: np.ndarray of the label test set

    """

    assert (X_data.shape[1] == Y_data.shape[1] ), "The number of pixels in the input and label data should be the same."
    assert (Y_data.shape[0] == 1), "The label y-vector should be a row vector."


    num_pixels = X_data.shape[1]

    train_percent, dev_percent, test_percent = set_percent   #the indicies that mark the ending of the training, dev, and test sets

    assert( abs( train_percent + dev_percent + test_percent - 1) < 0.01 ) , "the percentage splits of the data should sum to around 100%"

    train_index, dev_index, test_index = int(train_percent * num_pixels), int(dev_percent * num_pixels), int(test_percent * num_pixels)

    # print below for debugging
    # print(train_index, dev_index, test_index)

    col_indicies = np.arange(num_pixels) # a numpy list of the columns indicies
    np.random.shuffle(col_indicies) # shuffles the indicies in col_index without returning anything

    
    X_train = X_data[:,col_indicies[:train_index]]
    Y_train = Y_data[:,col_indicies[:train_index]]

    # rather than an index, the dev_index is the number of pixels in the dev-set, so it must be added to the train_index to properly index col_indicies
    X_dev = X_data[:, col_indicies[train_index: dev_index+train_index]]
    Y_dev = Y_data[:, col_indicies[train_index: dev_index+train_index]]

    # the same is true with the test_index as with the dev_index
    X_test = X_data[:, col_indicies[dev_index+train_index: dev_index+train_index+test_index]]
    Y_test = Y_data[:, col_indicies[dev_index+train_index: dev_index+train_index+test_index]]

    # print statements below for debugging
    # print(f"X_data shape: {X_data.shape}")
    # print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    # print(f"X_dev shape: {X_dev.shape}, Y_dev shape: {Y_dev.shape}")
    # print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")


    assert X_train.shape[1] + X_dev.shape[1] + X_test.shape[1] == X_data.shape[1], "The data divisions should sum to the total number of data points"
    assert Y_train.shape[1] + Y_dev.shape[1] + Y_test.shape[1] == Y_data.shape[1], "The data divisions should sum to the total number of data points"

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test



def mini_batches(X, Y, mini_batch_size = 256, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


