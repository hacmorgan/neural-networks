# Script collating the code from chapter 1

# NOTE: This is python2 code. We will attempt to run it in python3, but not make any changes that would break python2 compatibility in case it doesn't work

# Imports
import numpy as np
import random

# Network class
class network(object):
    
    # Constructor function: initialises biases and weights randomly
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        """
        Suppose sizes = [2,3,1]:
        Then sizes[:-1] = [2,3] and sizes[1:] = [3,1]
        So zip(sizes[:-1], sizes[1:]) = [(2,3), (3,1)]
        This is required for weights as it gives the correct number of elements for each node to connect to all nodes in the next layer
        """

    # Sigmoid function
    def sigmoid(z):
        # Note that the np.exp function applies this elementwise if a vector is supplied
        return 1.0/(1.0+np.exp(-z))

    # Feedforward function returning the network output for network input ndarray, a
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    # Function to perform stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        This trains the neural network using mini-batch stochastic gradient descent.
        
        training_data is a list of tuples (x,y) representing training inputs and desired outputs respectively.
        The other inputs are self explanatory, see the notes document for their definitions

        If test_data is provided, the network will be evaluated against the test data after each epoch, and partial progress printed out. This helps to track progress but reduces the speed of the algorithm.
        """

        if test_data:
            n_test = len(test_data)
            n = len(training_data)

        # Note that xrange is not a python3 function, as python3's range is python2's xrange
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using back propagation to a single mini-batch.
        The mini-batch is a list of tuples (x,y), and eta, the learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]
