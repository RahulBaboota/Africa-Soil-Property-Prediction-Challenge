import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two hidden layer fully-connected neural network. The net has an input dimension of
    N, hidden layer dimensions of H1 and H2, and it performs classification over C classes.
    The network is trained with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    The network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax layer

    The outputs of the third fully-connected layer are the raw scores for each class.
    """

    def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize, std = 1e-4):

        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        Inputs:
        - inputSize: The dimension D of the input data.
        - hiddenSize1: The number of neurons H1 in the hidden layer.
        - hiddenSize2: The number of neurons H2 in the hidden layer.
        - outputSize: The number of classes C.
        """

        self.params = {}
        self.params['W1'] = std * np.random.randn(inputSize, hiddenSize1)
        self.params['b1'] = np.zeros(hiddenSize1)
        self.params['W2'] = std * np.random.randn(hiddenSize1, hiddenSize2)
        self.params['b2'] = np.zeros(hiddenSize2)
        self.params['W3'] = std * np.random.randn(hiddenSize2, outputSize)
        self.params['b3'] = np.zeros(outputSize)

    def loss(self, X, y = None, reg = 0.0):
        """
        Compute the loss and gradients for the two hidden layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i].
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C).

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function.
        """

        #############################################################################
        #                              Forward Pass                                 #
        #############################################################################

        ## Unpack variables from the params dictionary.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass.
        scores = None

        ## Computing the first hidden layer.
        hiddenLayer1 = X.dot(W1) + b1

        ## Applying Relu to the hidden layer.
        activatedHiddenLayer1 = np.clip(hiddenLayer1, 0, None)

        ## Computing the second hidden layer.
        hiddenLayer2 = activatedHiddenLayer1.dot(W2) + b2

        ## Applying Relu to the hidden layer.
        activatedHiddenLayer2 = np.clip(hiddenLayer2, 0, None)

        ## Computing the final fully connected layer.
        scores = activatedHiddenLayer2.dot(W3) + b3

        ## If y is not given, simply return the scores.
        if y is None:
          return scores

        #############################################################################
        #                              Compute Loss                                 #
        #############################################################################

        ## Compute the loss.
        loss = None

        ## Normalise the raw scores to avoid exponential score blow-up.
        ## To do so, subtract the maximum score from each score value for each image.
        expScores = np.exp(scores - np.max(scores, axis = 1, keepdims = True))

        ## Compute the probabilities (or softmax scores) of each class.
        softmaxScores = expScores/np.sum(expScores, axis = 1, keepdims = True)

        ## Creating a 1-D matrix containing the softmax score of the correct class.
        corrSoftScore = np.choose(y, softmaxScores.T)

        ## Computing the cross-entropy loss.
        loss = -np.sum(np.log(corrSoftScore), axis = 0, keepdims = True)

        ## Compute the full training loss by dividing the cummulative loss by the number of training instances.
        loss = loss[0]/N

        ## Add regularisation loss.
        loss = loss + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)

        #############################################################################
        #                              Backward Pass                                #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}

        # Backward pass: compute gradients
        dO = softmaxScores

        ## Computing dL/dO (Softmax Gradient).
        dO[np.arange(N), y] -= 1
        dO /= N

        ## Computing dL/db3.
        grads['b3'] = np.sum(dO, axis = 0)

        ## Computing dL/dW3.
        grads['W3']= activatedHiddenLayer2.T.dot(dO) + reg * W3

        ## Computing dL/dActivatedHiddenLayer2.
        dActivatedHiddenLayer2 = dO.dot(W3.T)

        ## Computing dL/dHiddenLayer2 (Backprop through Relu).
        dActivatedHiddenLayer2[activatedHiddenLayer2 <= 0] = 0

        ## Computing dL/db2.
        grads['b2'] = np.sum(dActivatedHiddenLayer2, axis = 0)

        ## Computing dL/dW2.
        grads['W2'] = activatedHiddenLayer1.T.dot(dActivatedHiddenLayer2) + reg * W2

        ## Computing dL/dActivatedHiddenLayer1.
        dActivatedHiddenLayer1 = dActivatedHiddenLayer2.dot(W2.T)

        ## Computing dL/dHiddenLayer2 (Backprop through Relu).
        dActivatedHiddenLayer1[activatedHiddenLayer1 <= 0] = 0

        ## Computing dL/db1.
        grads['b1'] = np.sum(dActivatedHiddenLayer1, axis = 0)

        ## Computing dL/dW1.
        grads['W1'] = X.T.dot(dActivatedHiddenLayer1) + reg * W1

        return loss, grads

    def train(self, X, y, learningRate = 1e-3, learningRateDecay = 0.95,
            reg = 1e-5, numIters = 100, batchSize = 128, verbose = False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels.
        - learningRate: Scalar giving learning rate for optimization.
        - learningRateDecay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - numIters: Number of steps to take when optimizing.
        - batchSize: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        numTrain = X.shape[0]
        iterations_per_epoch = max(numTrain / batchSize, 1)

        # Use SGD to optimize the parameters.
        lossHistory = []
        trainAccHistory = []

        for it in xrange(numIters):

          XBatch = None
          yBatch = None

          ## Creating an array which randomly selects images.
          randomIndices = np.random.choice(np.arange(numTrain), size = batchSize)
          XBatch = X[randomIndices]
          yBatch = y[randomIndices]

          ## Compute loss and gradients using the current minibatch.
          loss, grads = self.loss(X = XBatch, y = yBatch, reg = reg)
          lossHistory.append(loss)

          ## Updating the weights and biases using stochastic gradient descent.

          self.params['W3'] -= learningRate * grads['W3']
          self.params['b3'] -= learningRate * grads['b3']

          self.params['W2'] -= learningRate * grads['W2']
          self.params['b2'] -= learningRate * grads['b2']

          self.params['W1'] -= learningRate * grads['W1']
          self.params['b1'] -= learningRate * grads['b1']

          ## Training Progress Log.
          if verbose and it % 100 == 0:
            print 'iteration %d / %d: loss %f' % (it, numIters, loss)

          ## After every epoch, check training accuracy and decay the learning rate.
          if it % iterations_per_epoch == 0:

            # Check accuracy
            trainAcc = (self.predict(XBatch) == yBatch).mean()
            trainAccHistory.append(trainAcc)

            # Decay learning rate
            learningRate *= learningRateDecay

        return {
          'lossHistory': lossHistory,
          'trainAccHistory': trainAccHistory,
        }

    def predict(self, X):
        """
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - yPred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X.
        """
        yPred = None

        ## Unpack variables from the params dictionary.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        ## Computing the first hidden layer.
        hiddenLayer1 = X.dot(W1) + b1

        ## Applying Relu to the hidden layer.
        activatedHiddenLayer1 = np.clip(hiddenLayer1, 0, None)

        ## Computing the second hidden layer.
        hiddenLayer2 = activatedHiddenLayer1.dot(W2) + b2

        ## Applying Relu to the hidden layer.
        activatedHiddenLayer2 = np.clip(hiddenLayer2, 0, None)

        ## Computing the final fully connected layer.
        scores = activatedHiddenLayer2.dot(W3) + b3

        ## Creating a vector containing the class prediction.
        yPred = np.argmax(scores, axis = 1)

        return yPred