import tensorflow as tf
import numpy as np




def create_placeholders(n_x, n_y):
    '''
    Description:
    This creates the placeholders of the tensorflow session.
    
    Inputs:
    n_x -- a scalar of the image size ( pixels)
    n_y -- a scalar of the number of classes
    
    Outputs:
    X -- a tf placeholder for the input data of shape [n_x, None] and data type float32
    Y -- a tf placeholder for the input labels of shape [n_y, None] and data type float32
    '''
    
    X = tf.placeholder(dtype=tf.float32, shape = [n_x, None], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape = [n_y, None], name='Y')
    
    return X, Y