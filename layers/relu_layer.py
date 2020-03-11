""" ReLU Layer """

import numpy as np


class ReLULayer():
    def __init__(self):
        """
        Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
        """
        self.trainable = False  # no parameters

    def forward(self, Input):
        ############################################################################

        self.x = Input
        return np.maximum(0, Input)

        ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        return delta * (self.x > 0)

        ############################################################################


if __name__ == '__main__':
    # correctness test
    x = np.array([[2, 2], [-1, -3]])
    relu = ReLULayer()
    y = relu.forward(x)
    test = np.array([[2, 2], [0, 0]])
    if (y == test).all():
        print("Test passed!")
    else:
        print("Test failed!")
    z = np.array([[1, 1], [1, 1]])
    y = relu.backward(z)
    test = np.array([[1, 1], [0, 0]])
    if (y == test).all():
        print("Test passed!")
    else:
        print("Test failed!")
