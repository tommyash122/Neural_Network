import Neural_Network
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

# This main module represent the main module from which the program starts and contain
# the functionally needed to work with the given database.
#
# After the system finishes training it creates a user interface where the user can enter
# some iris features and watch the system guess which type of iris is it, based on its knowledge.

# Basic variables definition
EPOCHS = 200  # number of cycles
LEARNING_RATE = 0.15
INPUT_LEN = 4  # number of features
OUTPUT_LEN = 3  # number of classes
HIDDEN_LAYER1 = 5  # number of nodes in hidden layer 1
LAYERS = [INPUT_LEN, HIDDEN_LAYER1, OUTPUT_LEN]  # number of nodes in each layer
CLASSES = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']

TRAINING_GROUP_ERROR_DATA = []
TESTING_GROUP_ERROR_DATA = []
MISTAKES_DATA = [
    "      Input       = Expected output , Network's guess  | ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']\n"
    "#############################################################################################################"]


def main():
    iris = datasets.load_iris()
    Y = np.asarray(iris['target'])  # list of answers of the features of the iris flowers : 0,1,2
    # transform the targets 0,1,2 into np arrays : 0 = [1. 0. 0.] , 1 = [0. 1. 0.] , 2 = [0. 0. 1.]
    Y = OneHotEncoder(sparse_output=False).fit_transform(np.array(Y).reshape(-1, 1))
    X = np.asarray(iris['data'])  # list of features of the iris flowers : sepal length, sepal width ...

    X_train, Y_train, X_test, Y_test = data_split(X, Y, test_sz=0.2)

    print("Initialize our Neural Network")
    network = Neural_Network.NN(X_train, Y_train, X_test, Y_test)
    print("Start training process!")
    training_acc, testing_acc = 0, 0

    for e in range(1, EPOCHS + 1):
        network.train()  # start training our network, every iteration considered an Epoch

        # Edit the file mistakes_data who will provide as some useful info later on
        MISTAKES_DATA.append("_______________________________________________"
                             "\nEpoch {}/{}\n".format(e, EPOCHS))

        training_acc = accuracy(network, network.X_train, network.Y_train)
        testing_acc = accuracy(network, network.X_test, network.Y_test)

        # subtract the accuracy value from 1 will give us the error rate throw the Epoch cycle
        TRAINING_GROUP_ERROR_DATA.append(1 - training_acc)
        TESTING_GROUP_ERROR_DATA.append(1 - testing_acc)

        # multiply by 100 to present the accuracy rate in percentages form
        training_acc, testing_acc = training_acc * 100, testing_acc * 100

        if e <= 10:
            print("Epoch [{}] -> Training Accuracy : [{:.2f}%]  Testing Accuracy : [{:.2f}%]"
                  .format(e, training_acc, testing_acc))

        elif e % 20 == 0:
            print("Epoch [{}] -> Training Accuracy : [{:.2f}%]  Testing Accuracy : [{:.2f}%]"
                  .format(e, training_acc, testing_acc))

        network.update_data(X, Y, test_sz=0.2)
    print("Neural Network has finished training.")
    print("Current Test Accuracy Rate: {:.2f}%. \n\n".format(testing_acc))

    user_input(network)


# The user interface with the system, insert some data and the system will provide an answer
def user_input(network):
    print("Enter 4 IRIS FLOWER DATA SET feature values, each separated by SPACE, for the system to recognize\n"
          "The values entered by the order of [sepal length, sepal width, petal length, petal width]"
          " within the range of 0 to 8")
    print("If you wish to QUIT type: exit ")

    while True:
        _str = input(" > ")
        _str = _str.strip()  # used for striping the input from prefix and postfix redundant spaces

        if _str.__eq__("exit"):
            print("Good bye!")
            return

        # input validation
        if not _str:
            continue

        str_array = _str.split(' ')
        if not len(str_array) == 4:
            print("INVALID INPUT:please try to enter exactly 4 values")
            continue

        for v in str_array:
            if not check_user_input(v):
                print("INVALID INPUT:please try to enter only number values")
                continue

            if not 0 <= float(v) <= 8:
                print(
                    "INVALID INPUT:please try to enter only proportional values according to the range of values given")
                continue

        if sum(np.asarray(str_array, dtype=float)) < 4.3:  # the sepal is at least 4.3Cm
            print("INVALID INPUT:please try to enter only proportional values according to the range of values given")
            continue

        layer_input = np.asarray(str_array, dtype=float)
        print(layer_input)
        ans = predict(layer_input, network)

        for i in range(len(CLASSES)):
            if ans[i] == 1:
                print(CLASSES[i])


def check_user_input(v):
    try:
        # Convert it into integer
        int(v)
        return True
    except ValueError:
        try:
            # Convert it into float
            float(v)
            return True
        except ValueError:
            return False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.multiply(x, 1 - x)


# For the sake of study we will try to perform a run with a binary activation function
# we will test the results with those two activation functions and see the deference
# between those two and the sigmoid and sigmoid_derivative
###############################################################################
def binary_step(x):
    # if x < 0 return 0 else return 1
    return np.heaviside(x, 1)


def binary_step_derivative(x):
    # if x == 0 undefined so we will return 0 else act like binary_step
    return np.heaviside(x, 0)


###############################################################################


# Splits the data to a training and testing groups randomly
def data_split(X, Y, test_sz):
    inds = np.array(np.arange(0, len(X)))
    np.random.shuffle(inds)  # mix the numbers

    start = 0
    end = int((len(X) * test_sz))

    test_inds = inds[start:end]  # the test group indices
    train_inds = inds[end:len(X)]  # the train group indices

    X_train = [X[i] for i in train_inds]  # adding up the values to the right array verbs
    Y_train = [Y[i] for i in train_inds]
    X_test = [X[i] for i in test_inds]
    Y_test = [Y[i] for i in test_inds]

    X_train = np.array(X_train)  # converting the arrays into np arrays
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test


# Test the accuracy level of our prediction by the current adjusted weights of our network
def accuracy(network, X, Y):
    correct_cnt = 0
    mistake_cnt = 0

    # runs on all the X list given, insert every line (input) from it one by one to the network,
    # gets an answer , compare this answer to the corresponding Y line (correct output)
    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = predict(x, network)

        if y == guess:
            correct_cnt += 1
        else:
            mistake_cnt += 1
            MISTAKES_DATA.append("{} = {} , {}".format(x, y, guess))

    # number of correct answers divided by the entire training group
    return correct_cnt / len(X)


# Push some input into our network and get a result
def predict(x, network):
    num_layers = len(network.weights)
    x = np.append(1, x)  # add 1 for the bias

    activations = network.forward_propagation(x, num_layers)
    output = np.ravel(activations[-1])
    index = find_max_act(output)

    y = [0 for _ in range(len(output))]
    y[index] = 1

    return y


def find_max_act(output):
    _max, index = output[0], 0

    for i in range(1, len(output)):
        if output[i] > _max:
            _max, index = output[i], i
    return index


if __name__ == "__main__":
    main()  # Launch the program

    file1 = open("Epoch_Training_Data.txt", 'w')
    for i in TRAINING_GROUP_ERROR_DATA:
        file1.write("{}\n".format(i))
    file1.close()

    file2 = open("Epoch_Testing_Data.txt", 'w')
    for i in TESTING_GROUP_ERROR_DATA:
        file2.write("{}\n".format(i))
    file2.close()

    file3 = open("Mistakes_Data.txt", 'w')
    for i in MISTAKES_DATA:
        file3.write("{}\n".format(i))
    file2.close()
