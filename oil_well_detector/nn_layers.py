import tensorflow as tf
import numpy as np
from typing import Tuple, List





def create_placeholders(n_x: int, n_y: int) ->Tuple[tf.placeholder, tf.placeholder]:
    """This function creates the placeholders of the tensorflow session.
    
    Parameters
    ----------
    n_x: an integer
        the number of bands or images imputted
    n_y: an integer
        the number of classes
    
    Returns
    ---------
    X: tf.placeholder of shape [n_x, None] and data type float32
        a tf placeholder for the input data 
    Y: tf.placeholder of shape [n_y, None] and data type float32
        a tf placeholder for the input labels 

    """
    
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape = [n_x, None], name='X')
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape = [n_y, None], name='Y')
    
    return X, Y


def initialize(layers_dims: List[int], initialization: str = 'xavier'):
    '''Initializes the parameters in neural network based on the size of layers_dims where each value of layers_dims 
        specifies the number of neurons in each layer. The 0-th value of layers_dims is the size of the input data, A0.
        
    Parameters
    -----------
    layers_dims: list of integers of shape (1 x number of layers + 1)
        a list that defines the number of layers and the number of hidden units in each layer 
        
    initialization: string
        a keyword arguement that defines the kind of initialization for the variables. The two options are:
            'xavier' for xavier initialization 
            'randn' for random normal.
        
   Returns
   -----------
    parameters: dict object
        a dictionary of the intialize parameters for the nueral network
    
    '''
    
    ld = layers_dims # an abbreviation for layers_dims
    parameters = {}
    
    if initialization == 'xavier':
        for l in range(1, len(layers_dims)):
            parameters['W'+str(l)] = tf.get_variable('W'+str(l), [ld[l], ld[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            parameters['b'+str(l)] = tf.get_variable('b'+str(l), [ld[l], 1], initializer = tf.zeros_initializer())
    
    elif initialization == 'randn':
        for l in range(1, len(layers_dims)):
            parameters['W'+str(l)] = tf.get_variable('W'+str(l), [ld[l], ld[l-1]], initializer = tf.initializers.random_normal(seed = 1))*0.01
            parameters['b'+str(l)] = tf.get_variable('b'+str(l), [ld[l], 1], initializer = tf.zeros_initializer())
    
    else:
        raise ValueError("Initialization keyword not recognized")

    return parameters