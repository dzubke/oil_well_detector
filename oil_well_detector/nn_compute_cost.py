import tensorflow as tf

def compute_cost(Z_L, Y):
    '''Creates a tensor of sigmoid cost function
    
    Parameters
    -----------
    Z_L  --  the linear unit of the last layer in the neural network, of shape (# classes (1) x # examples)
    Y    --  the input labels, of shape (1, # examples)
    
    Returns
    ----------
    cost -- the tensor of the cost function
    '''
    
    #transposes the inputs to fit the needed dimensions of the softmax cost function
    logits = tf.transpose(Z_L)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost