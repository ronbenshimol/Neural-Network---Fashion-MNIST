import sys
import numpy as np
import random
import copy  # for deep copy
import scipy

def getDefaultMatrix(rowsNum, columnsNum = None):
    if columnsNum != None:
        return np.random.uniform(-0.08, 0.08, (rowsNum, columnsNum))
    else:
        return np.random.uniform(-0.08, 0.08, rowsNum)

def negativeLogLikelihood(probabilityVector):
    return -np.log(probabilityVector)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork:

    def __init__(self, inputSize, outputSize, activationFunc, learnRate):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activationFunc = activationFunc
        self.learnRate = learnRate

        hiddenLayerSize = 16

        # creating 2 hidden layers:
        # init first hidden layer to 16 neurons (matrix[16][inputSize])
        self.w1 = getDefaultMatrix(hiddenLayerSize,inputSize)
        # init first hidden layer vector of bias
        self.b1 = getDefaultMatrix(hiddenLayerSize)

        # init second hidden layer to 16 neurons (matrix[outputSize][16])
        self.w2 = getDefaultMatrix(outputSize,hiddenLayerSize)
        # init second hidden layer vector of bias
        self.b2 = getDefaultMatrix(outputSize)

        # grouping the layers
        self.weights = [self.w1,self.w2]
        self.biases = [self.b1,self.b2]
    
    def feedForward(self, x):
        ''' x: input data set '''
        z = np.dot(self.w1, x) + self.b1
        # h is the first hidden layer output after activation
        h = self.activationFunc(z)
        z2 = np.dot(self.w2, h) + self.b2
        predictedVector = softmax(z2)
        return predictedVector, h

    def backPropagation(self, x, y, predictedVector, h):
        # calculating the new weights using the chain rule
        funcW2 = np.outer(predictedVector, h)
        funcW2[y, :] = funcW2[y, :] - h
        funcB2 = np.copy(predictedVector) - 1

        z1 = np.dot(self.w1, x) + self.b1
        # calc the dLoss / dFunc
        dLoss_dFunc = np.dot(predictedVector, self.w2)
        dLoss_dFunc -= self.w2[y, :]

        # calc the dFunc / dB1
        dFunc_dB1 = self.activationFunc(z1, True)

        funcB1 = dLoss_dFunc * dFunc_dB1
        funcW1 = np.outer(funcB1, x)

        prob_vec = predictedVector[y]
        loss = negativeLogLikelihood(prob_vec)
        return loss, funcW1, funcB1, funcW2, funcB2

    def updateNetwork(self, funcW1, funcB1, funcW2, funcB2):
        self.w1 -= self.learnRate * funcW1
        self.b1 -=  self.learnRate * funcB1
        self.w2 -=  self.learnRate * funcW2
        self.b2 -=  self.learnRate * funcB2
