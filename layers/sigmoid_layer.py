""" Sigmoid Layer """

import numpy as np


class SigmoidLayer():
    def __init__(self):
        """
        Applies the element-wise function: f(x) = 1/(1+exp(-x))
        """
        self.trainable = False

    def forward(self, Input):
        ############################################################################
        # TODO: Put your code here
        # Apply Sigmoid activation function to Input, and return results.

        self.Input = Input

        exp_scores = 1 / (1 + np.exp(-Input))
        return exp_scores

    ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        exp_scores = 1 / (1 + np.exp(-self.Input))
        return delta * exp_scores * (1 - exp_scores)

        ############################################################################


if __name__ == '__main__':
    # correctness test
    x = np.random.uniform(size=(10, 784))
    fc = SigmoidLayer()
    y = fc.forward(x)
    if y.shape == x.shape:
        print("Test passed!")
    else:
        print("Test failed!")
    d = np.random.uniform(size=(10, 784))
    y = fc.backward(d)
    if y.shape == x.shape:
        print("Test passed!")
    else:
        print("Test failed!")


