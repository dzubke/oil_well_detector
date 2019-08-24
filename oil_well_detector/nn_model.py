import tensorflow as tf
from tensorflow.python.framework import ops
import prep_data, nn_layers, nn_forward_prop, nn_compute_cost, explore_data
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def model(layers_dims: List[float], X_train, Y_train, X_test, Y_test, learning_rate: float = 0.05, 
            num_epochs: int = 50, minibatch_size: int = 64, print_cost: bool =True) -> Tuple[dict, np.ndarray]:
    '''The neural network optimization model
    
    Parameters
    -----------
    iternations: integer
        nummber of iterations of the optimization algorithm
    print_cost: boolean
        if True the algorithm will print the cost over a certain number of iterations
    
    Returns
    -----------
    parameters: dict
        dictionary of the trained parameters from the neural network
    
    
    Note1: Incorporate mini-batches into optimization
    '''
    
    # the code below was based on an exercise from the Deeplearning.ai course Improving Neural Networks on Coursera

    
    ops.reset_default_graph()           # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)               # to keep consistent results
    seed = 3                            # to keep consistent results
    (n_x, m) = X_train.shape            # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]              # n_y : output size
    costs = []                          # To keep track of the cost
    
    X, Y = nn_layers.create_placeholders(n_x, n_y)
    
    parameters = nn_layers.initialize(layers_dims)
    
    Z_L, _ = nn_forward_prop.forward_prop(X, parameters)
    
    cost = nn_compute_cost.compute_cost(Z_L, Y)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = prep_data.mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 2 == 0:
                print (f"Cost after epoch {epoch}: {epoch_cost}")
            if print_cost == True and epoch % 2 == 0:
                costs.append(epoch_cost)
                
        """
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        """

        parameters = sess.run(parameters)
        Z_train = Z_L.eval({X:X_train, Y:Y_train})
        Z_test = Z_L.eval({X:X_test, Y:Y_test})

        correct_prediction = tf.equal(Z_L, Y)
        print(f"correct prediction is: {correct_prediction}")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(f"accuracy is: {accuracy}")
        print('Train Accuracy: ', accuracy.eval({X:X_train, Y:Y_train} ) )
        print('Test Accuracy: ', accuracy.eval({X:X_test, Y:Y_test} ) )
       
        """ this code isn't working
        predictions = tf.compat.v2.math.sigmoid(Z_L)
        f1_score, update_op = tf.contrib.metrics.f1_score(Y, predictions)
        print(f"F1 score for train set is: {f1_score.eval({X:X_train, Y:Y_train})}")
        print(f"F1 score for test set is: {f1_score.eval({X:X_test, Y:Y_test})}")
        """
        return parameters, Z_train, Z_test