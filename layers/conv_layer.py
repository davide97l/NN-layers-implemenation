# -*- encoding: utf-8 -*-

import numpy as np
from im2col import *


class ConvLayer():
    """
    2D convolutional layer.
    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs.
    Arguments:
        inputs: Integer, the channels number of input.
        filters: Integer, the number of filters in the convolution.
        kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
        pad: Integer, the size of padding area.
        trainable: Boolean, whether this layer is trainable.
    """

    def __init__(self, inputs,
                 filters,
                 kernel_size,
                 pad=0,
                 trainable=True,
                 stride=1):
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.pad = pad
        assert pad < kernel_size, "pad should be less than kernel_size"
        self.trainable = trainable
        self.stride = stride

        self.XavierInit()

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def XavierInit(self):
        raw_std = (2 / (self.inputs + self.filters)) ** 0.5
        init_std = raw_std * (2 ** 0.5)

        self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
        self.b = np.random.normal(0, init_std, (self.filters, ))

    def forward(self, Input):
        '''
        forward method: perform convolution operation on the input.
        Args:
            Input: A batch of images, shape-(batch_size, channels, height, width)
        '''
        ############################################################################

        self.Input = Input

        n_x, d_x, h_x, w_x = Input.shape
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        h_out = (h_x - h_filter + 2 * self.pad) // self.stride + 1  # feature map height
        w_out = (w_x - w_filter + 2 * self.pad) // self.stride + 1  # feature map length

        # stretch the input batch of images
        self.X_col = im2col_indices(Input, self.kernel_size, self.kernel_size, padding=self.pad, stride=self.stride)
        # reshape weights
        W_col = self.W.reshape(self.filters, -1)
        # multiply inputs times weights and add bias
        out = W_col @ self.X_col + self.b.reshape(self.filters, 1)

        # reshape back from (filters x (h_out x w_out x n_x)) to (n_x x filters x h_out x w_out)
        out = out.reshape(self.filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        return out

        ############################################################################

    def backward(self, delta):
        '''
        backward method: perform back-propagation operation on weights and biases.
        Args:
            delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
        Return:
            delta of previous layer
        '''
        ############################################################################
        # Calculate self.grad_W, self.grad_b, and return the new delta.

        # the bias is added to each of our filter, so we’re accumulating the gradient to the dimension
        # that represent of the number of filter, which is the second dimension.
        # Hence the sum is operated on all axis except the second.
        db = np.sum(delta, axis=(0, 2, 3))
        self.grad_b = db.reshape(self.filters)

        # Reshape from (n_x x filters x h_out x w_out) to (filters x (h_out x w_out x n_x))
        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        # use input to make convolution on the gradient
        dW = delta_reshaped @ self.X_col.T
        # Reshape back to the shape of the weight matrix
        self.grad_W = dW.reshape(self.W.shape)

        # Reshape from (filters x 1 x kernel_size x kernel_size) into (filters x (kernel_size x kernel_size))
        W_reshape = self.W.reshape(self.filters, -1)
        # use weights to make convolution on the gradient
        dX_col = W_reshape.T @ delta_reshaped
        # Stretched out image to the real image
        dX = col2im_indices(dX_col, self.Input.shape, self.kernel_size, self.kernel_size,
                            padding=self.pad, stride=self.stride)
        # gradient respect to the input
        return dX

        ############################################################################


if __name__ == '__main__':
    # correctness test
    x = np.random.uniform(size=(1, 1, 10, 10))
    m = im2col_indices(x, 3, 3, 1)
    test_shape = np.array([9, 100])
    if (m.shape == test_shape).all():
        print("Test passed!")
    else:
        print("Test failed!")
    x1 = col2im_indices(m, x.shape, 3, 3)
    if x1.shape == x.shape:
        print("Test passed!")
    else:
        print("Test failed!")
    cl = ConvLayer(1, 3, 3, 1)
    res = cl.forward(x)
    test_shape = np.array([1, 3, 10, 10])
    if (res.shape == test_shape).all():
        print("Test passed!")
    else:
        print("Test failed!")
    cl = ConvLayer(1, 7, 5, 0)
    res = cl.forward(x)
    test_shape = np.array([1, 7, 6, 6])
    if (res.shape == test_shape).all():
        print("Test passed!")
    else:
        print("Test failed!")
    res1 = cl.backward(res)
    if x.shape == res1.shape:
        print("Test passed!")
    else:
        print("Test failed!")
