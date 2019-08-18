import sys
import numpy as np
import random
import copy  # for deep copy
import scipy
from NeuralNetwork import NeuralNetwork

def shuffle(x, y):
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y

def sigmoid(x, derivative=False):
    if (derivative == True):
        return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))
    return 1 / (1 + np.exp(-x))

def negativeLogLikelihood(probabilityVector):
    return -np.log(probabilityVector)



def train(neuralNetwork, trainSet, labelSet, epochs):
    
    for epochsNum in range(epochs):

        trainSetShuffled, labelSetShuffled = shuffle(trainSet, labelSet)
        for x, y in zip(trainSetShuffled, labelSetShuffled):
            # convert to int
            y = int(y)
            # feedforward
            predictedVector, h = neuralNetwork.feedForward(x)
            # back propagation
            loss, funcW1, funcB1, funcW2, funcB2 = neuralNetwork.backPropagation(x, y, predictedVector, h)
            neuralNetwork.updateNetwork(funcW1, funcB1, funcW2, funcB2)




def test(network, valdiation_x_dataset, valdiation_y_dataset):
    sum_loss = 0.0
    success = 0

    # shuffle validation set (again to make more accurate)
    x_vec, y_vec = shuffle(valdiation_x_dataset, valdiation_y_dataset)

    for x, y in zip(x_vec, y_vec):
        y = int(y)  # set as int since its only the tag
        predicted_vec, h = network.feedForward(x)
        prob_vec = predicted_vec[y]
        loss = negativeLogLikelihood(prob_vec)
        # loss = -np.log(predicted_vec[y])
        sum_loss = sum_loss + loss
        y_hat = predicted_vec.argmax()
        if y == y_hat:
            success += 1

    validation_vec_size = float(valdiation_x_dataset.shape[0])
    avg_loss = sum_loss / validation_vec_size
    accuracy = success / validation_vec_size

    return avg_loss, accuracy

def main():
    inputSize = 784
    outputSize = 10
    trainRatio = 0.8
    learnRate = 0.02
    epochs = 16
    activationFunction = sigmoid

    trainX = np.loadtxt("train_x")
    trainY = np.loadtxt("train_y")
    # TODO: uncomment
    # testX = np.loadtxt("test_x")
    np.random.seed(2)

    shuffledTrainX,shuffledTrainY = shuffle(trainX,trainY)
    # creating training set by spliting the dataset into two parts
    train_size = int(trainRatio * shuffledTrainX.shape[0])
    shuffledTrainSet = shuffledTrainX[:train_size]
    shuffledLabelSet = shuffledTrainY[:train_size]
    validationX = shuffledTrainX[train_size:]
    validationY = shuffledTrainY[train_size:]
    
    # normalize to values to be between 0 to 1
    shuffledTrainSet /= 255.0
    validationX /= 255.0
    
    for epochs in range(33):

        np.random.seed(2)

        # creating instance of the neural network
        network = NeuralNetwork(inputSize, outputSize, activationFunction, learnRate)

        train(network,copy.deepcopy(shuffledTrainSet), copy.deepcopy(shuffledLabelSet), epochs)

        avg_loss, accuracy = test(network, validationX, validationY)

        print("epochs: " + str(epochs) + " , avg_loss: " + str(avg_loss) +" , accuracy: " + str(accuracy) + "\n")


if __name__ == "__main__":
    main()