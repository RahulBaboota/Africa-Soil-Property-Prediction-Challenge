import numpy as np
from neuralNetwork import TwoLayerNet

## Initialising toy data and specifying details of the neural network.
inputSize = 4
hiddenSize1 = 5
hiddenSize2 = 5
outputSize = 3
numInputs = 5

## Toy Data.
X = 10 * np.random.randn(numInputs, inputSize)
y = np.array([0, 1, 2, 2, 1])

## Train the neural Network.
net = TwoLayerNet(inputSize, hiddenSize1, hiddenSize2, outputSize)
stats = net.train(X, y, learningRate=1e-1, reg=1e-5, numIters=5000)

## Perform the forward pass.
yPred = net.predict(X)