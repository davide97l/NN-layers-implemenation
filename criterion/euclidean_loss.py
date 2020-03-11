""" Euclidean Loss Layer """

import numpy as np


class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.

    def forward(self, logit, gt):
        """
          Inputs: (minibatch)
          - logit: forward results from the last FCLayer, shape(batch_size, 10)
          - gt: the ground truth label, shape(batch_size, 10)
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.

        self.logit = logit
        self.gt = gt

        num_trains = logit.shape[0]
        predictions = np.argmax(logit, axis=1)
        real_values = np.argmax(gt, axis=1)
        self.acc = np.sum(predictions == real_values) / num_trains

        diff = predictions - real_values
        self.loss = 0.5 * np.sum((diff ** 2), axis=-1)

        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return self.logit - self.gt

        ############################################################################


if __name__ == '__main__':
    # correctness test
    x = np.random.uniform(size=(10, 784))
    gt = np.random.uniform(size=(10, 1))
    el = EuclideanLossLayer()
    y = el.forward(x, gt)
    y = el.backward()
    test = np.random.uniform(size=10)
    if y.shape == x.shape:
        print("Test passed!")
    else:
        print("Test failed!")

