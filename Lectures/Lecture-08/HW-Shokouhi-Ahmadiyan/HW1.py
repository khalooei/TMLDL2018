import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        # self.inputLayerSize = 2
        # self.outputLayerSize = 1
        # self.hiddenLayerSize = 3
        self.inputLayerSize = 784
        self.outputLayerSize = 1
        self.hiddenLayerSize = 100000

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': False}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='L-BFGS-B', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

import  mnist
if __name__ == '__main__':
    x_train, t_train, x_test, t_test = mnist.load()
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    # # X = (hours sleeping, hours studying), y = Score on test
    # X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    # y = np.array(([75], [82], [93]), dtype=float)
    # # Normalize
    # X = X / np.amax(X, axis=0)
    # y = y / 100  # Max test score is 100

    train_number=100
    X =x_train[0:train_number,:]
    y=t_train[0:train_number]
    y=y.reshape((len(y),1))

    NN = Neural_Network()

    print('y is ',y[0:10])
    yHat=NN.forward(X)
    print('yHat before train ',yHat[0:10])

    T = trainer(NN)
    T.train(X, y)
    yHat = np.round(NN.forward(X))
    print('yHat after is ',yHat[0:10])

    # # plt.plot(T.J)
    # # plt.grid(1)
    # # plt.xlabel('Iterations')
    # # plt.ylabel('Cost')
    # # plt.show()