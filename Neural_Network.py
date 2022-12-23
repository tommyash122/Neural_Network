import main
import numpy as np


# This class is a represents a Neural Network with one input layer, one output layer and one hidden layer.
# Every layer's node connected the all the nodes in the following layer except for output layer nodes.
# All nodes share edges with weights on each one that will be adjusted properly according to the given database.

class NN:  # Neural Network
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.weights = self.init_weights()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    # update our train and test groups
    def update_data(self, X, Y, test_sz):
        self.X_train, self.Y_train, self.X_test, self.Y_test = main.data_split(X, Y, test_sz)

    # A function that uses to put forward our inputs through our network
    def forward_propagation(self, x, num_layers):
        activations, layer_input = [x], x

        for i in range(num_layers):
            # multiply our node values(input) by the weights of the edges to calculate the next node's value
            # the sigmoid of each dot product of each row on our weights matrix with the nodes values give us
            # the value of the following node on the next layer
            # Lines 30,55 is for the experiment, mentioned in main.py line
            # act = main.binary_step(np.dot(layer_input, self.weights[i].T))
            act = main.sigmoid(np.dot(layer_input, self.weights[i].T))
            activations.append(act)  # add the node new calculated value to our activations matrix
            layer_input = np.append(1, act)  # add 1 for bias

        return activations

    # A function that uses to update the weights of our network
    def back_propagation(self, y, activations, num_layers):
        output = activations[-1]  # the last element of our activations matrix is our output
        error = np.matrix(y - output)  # error after each cycle

        for i in range(num_layers, 0, -1):  # range(start, stop, step)
            currAct = activations[i]

            if i > 1:
                # Append the previous one
                prevAct = np.append(1, activations[i - 1])
            else:
                # first hidden layer
                prevAct = activations[0]

            # Calculate the derivative of our sigmoid function to indicate if our function going up or down.
            # In other words, to decide if we need to add or subtract
            # Lines 30,55 is for the experiment, mentioned in main.py line
            # derivAct = main.binary_step_derivative(currAct)
            derivAct = main.sigmoid_derivative(currAct)
            delta = np.multiply(error, derivAct)  # decides the amount of difference
            # add our new weight values with the known formula
            self.weights[i - 1] += main.LEARNING_RATE * np.multiply(delta.T, prevAct)

            # delete the old values and calculate the current error value
            wc = np.delete(self.weights[i - 1], [0], axis=1)
            error = np.dot(delta, wc)

    def train(self):
        num_layers = len(self.weights)

        for i in range(len(self.X_train)):  # Iterate throw all the training data group
            x, y = self.X_train[i], self.Y_train[i]  # the current input values and there answer
            x = np.matrix(np.append(1, x))  # add 1 as bias

            activations = self.forward_propagation(x, num_layers)
            self.back_propagation(y, activations, num_layers)

    # Initialize the weights matrix according the number of nodes in each layer with respect to the number of layers
    @staticmethod
    def init_weights():
        layers, weights = len(main.LAYERS), []

        for i in range(1, layers):
            w = []
            for j in range(main.LAYERS[i]):
                _ = []
                for k in range(main.LAYERS[i - 1] + 1):
                    _.append(np.random.uniform(-1, 1))
                w.append(_)
            weights.append(np.matrix(w))

        return weights
