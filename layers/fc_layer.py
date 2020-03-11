""" Fully Connected Layer """

import numpy as np


class FCLayer():
    def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            actFunction: the name of the activation function such as 'relu', 'sigmoid'
            trainable: whether if this layer is trainable
        """
        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.actFunction = actFunction
        assert actFunction in ['relu', 'sigmoid']

        self.XavierInit()

        self.grad_W = np.zeros((num_input, num_output))
        self.grad_b = np.zeros((1, num_output))

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation(Wx+b) to Input, and return results.

        self.x = Input
        return np.dot(Input, self.W) + self.b

        ############################################################################

    def backward(self, delta):

        # The delta of this layer has been calculated in the later layer.
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        N = self.x.shape[0]
        dx = np.dot(delta, self.W.T)
        dx.reshape(self.x.shape)

        dw = np.dot(self.x.T, delta)

        db = np.sum(delta, axis=0).reshape(1, self.num_output)

        self.grad_b = db
        self.grad_W = dw

        return dx

        ############################################################################

    def XavierInit(self):
        # Initialize the weigths according to the type of activation function.
        raw_std = (2 / (self.num_input + self.num_output)) ** 0.5
        if 'relu' == self.actFunction:
            init_std = raw_std * (2 ** 0.5)
        elif 'sigmoid' == self.actFunction:
            init_std = raw_std
        else:
            init_std = raw_std  # * 4

        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))


if __name__ == '__main__':
    # correctness test
    x = np.random.uniform(size=(10, 784))
    fc = FCLayer(784, 100)
    y = fc.forward(x)
    test_shape = np.array([10, 100])
    if (y.shape == test_shape).all():
        print("Test passed!")
    else:
        print("Test failed!")
    delta = np.random.uniform(size=(10, 100))
    dx, dw, db = fc.backward(delta)
    if dx.shape == x.shape:
        print("Test passed!")
    else:
        print("Test failed!")
    if dw.shape == fc.W.shape:
        print("Test passed!")
    else:
        print("Test failed!")
    if db.shape == fc.b.shape:
        print("Test passed!")
    else:
        print("Test failed!")
