""" Dropout Layer """

import numpy as np


class DropoutLayer():
    def __init__(self, p_dropout=0.5):
        self.trainable = False
        self.p = p_dropout

    def forward(self, Input, is_training=True):
        ############################################################################

        p_keep = 1 - self.p
        if not is_training:
            p_keep = 1
        self.Input = Input
        self.mask = np.random.binomial(1, p_keep, size=Input.shape)
        out = Input * self.mask
        if not is_training:
            out *= 1 - self.p
        return out

        ############################################################################

    def backward(self, delta):
        ############################################################################

        dX = delta * self.mask
        return dX

        ############################################################################


# correctness test
if __name__ == '__main__':
    x = np.ones([1, 1, 10, 10])
    dp = DropoutLayer(0.5)
    res = dp.forward(x, is_training=True)
    if (res != x).any():
        print("Test passed!")
    else:
        print("Test failed!")
    res = dp.forward(x, is_training=False)
    if (res == x * 0.5).all():
        print("Test passed!")
    else:
        print("Test failed!")
