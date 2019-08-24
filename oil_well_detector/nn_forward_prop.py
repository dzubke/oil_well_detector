
import tensorflow as tf
from typing import Tuple, List

def forward_prop(X: tf.placeholder, parameters:dict ) -> tf.Tensor:
    '''“Performs the forward propagation steps of the neural network using the relu activiation function
    
    Parameters
    -----------
    X:  tf.placeholder of shape (784 x # examples)
        the input data

    parameters: dict
        a dictionary of the parameters of the neural network, mainly W matricies and b vectors with keys 'W1' and 'b1' for the first layer
    
    Returns
    --------
    Z_L: tf.Tensor of shape (# classes (1 on this case) x # examples)
        the linear combination of the final layer, which will be fed into the tensorflow cost function
        
    '''
    
    
    cache = {'A0':X} #dictionary of the hidden linear and activation layers where the first A0 is X, the input data
    L = len(parameters)//2 # the number of layers in the neural network
    
    for l in range(1, L): #the loop stops at the L-1 layer because the final activiation layer
        W_l = parameters['W'+str(l)]
        b_l = parameters['b'+str(l)] #unpacking the W and b terms to make the linear combination more readable
        A_prev = cache['A'+str(l-1)] #the previous activation layer value. For l=1, A_prev = A0 = X
        cache['Z'+str(l)] = tf.add( tf.matmul(W_l, A_prev), b_l )
        cache['A'+str(l)] = tf.nn.relu(cache['Z'+str(l)])
    
    cache['Z'+str(L)] = tf.add( tf.matmul(parameters['W'+str(L)], cache['A'+str(L-1)]), parameters['b'+str(L)] )
    Z_L = cache['Z'+str(L)]
    
    return Z_L, cache