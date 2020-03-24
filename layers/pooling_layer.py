# -*- encoding: utf-8 -*-

import numpy as np
from im2col import *


class MaxPoolingLayer():
    def __init__(self, kernel_size, pad):
        '''
        This class performs max pooling operation on the input.
        Args:
            kernel_size: The height/width of the pooling kernel.
            pad: The width of the pad zone.
        '''

        self.kernel_size = kernel_size
        self.pad = pad
        self.trainable = False

    def forward(self, Input):
        '''
        This method performs max pooling operation on the input.
        Args:
            Input: The input need to be pooled.
        Return:
            The tensor after being pooled.
        '''
        ############################################################################

        self.Input = Input
        n, d, h, w = Input.shape

        h_out = (h - self.kernel_size + 2 * self.pad) // self.kernel_size + 1  # feature map height
        w_out = (w - self.kernel_size + 2 * self.pad) // self.kernel_size + 1  # feature map length

        # First, reshape it to ((n x d) x 1 x h x w) to make im2col arranges it fully in column
        Input_reshaped = Input.reshape(n * d, 1, h, w)

        # The result will be ((kernel_size**2) x number of convolutions)
        self.X_col = im2col_indices(Input_reshaped, self.kernel_size, self.kernel_size,
                                    padding=self.pad, stride=self.kernel_size)

        # Next, at each possible patch location, i.e. at each column, we're taking the max index
        self.max_idx = np.argmax(self.X_col, axis=0)

        # The result will be (1 x number of convolutions)
        out = self.X_col[self.max_idx, range(self.max_idx.size)]

        # Reshape to the output size: (feature map height x feature map length x batch_size x channels)
        out = out.reshape(h_out, w_out, n, d)

        # Transpose to get size: (batch_size x channels x feature map height x feature map length)
        out = out.transpose(2, 3, 0, 1)

        return out

    ############################################################################

    def backward(self, delta):
        '''
        Args:
            delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
        Return:
            delta of previous layer
        '''
        ############################################################################
        # Calculate and return the new delta.

        n, d, h, w = self.Input.shape

        # ((kernel_size**2) x number of convolutions), as in the forward step
        dX_col = np.zeros_like(self.X_col)

        # These operations are the opposite of the last operations of the forward pass
        delta_flat = delta.transpose(2, 3, 0, 1).ravel()

        # Fill the maximum index of each column with the gradient

        # Essentially putting each of the grads
        # to one of the (kernel_size**2) row in all locations, one at each column
        dX_col[self.max_idx, range(self.max_idx.size)] = delta_flat

        # We now have the stretched matrix of ((kernel_size**2) x number of convolutions),
        # then undo it with col2im operation, dX would be ((n x d) x 1 x h x w)
        dX = col2im_indices(dX_col, (n * d, 1, h, w), self.kernel_size, self.kernel_size,
                            padding=0, stride=self.kernel_size)

        # Reshape back to match the input dimension: 5x10x28x28
        dX = dX.reshape(self.Input.shape)

        return dX

        ############################################################################


# correctness test
if __name__ == '__main__':
    x = np.random.uniform(size=(1, 1, 10, 10))
    mp = MaxPoolingLayer(2, 0)
    res = mp.forward(x)
    test_shape = np.array([1, 1, 5, 5])
    if (res.shape == test_shape).all():
        print("Test passed!")
    else:
        print("Test failed!")
    x1 = mp.backward(res)
    if x.shape == x1.shape:
        print("Test passed!")
    else:
        print("Test failed!")
