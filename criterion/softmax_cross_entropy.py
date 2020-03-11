""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')

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

        num_train = logit.shape[0]
        y = np.where(self.gt == 1)[1]  # scalar value labels

        exp_scores = np.exp(logit)  # compute exp logits
        prob_scores = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + EPS)  # compute probabilities
        correct_log_probs = -np.log(prob_scores[range(num_train), y])  # compute loss for each row
        loss = np.sum(correct_log_probs)  # compute total loss
        self.loss = loss / num_train  # divide total loss by the number of rows

        prediction = np.argmax(exp_scores, axis=1)
        self.acc = sum(prediction == y) / num_train

        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)

        # compute gradient
        y = np.where(self.gt == 1)[1]  # scalar value labels
        num_train = self.logit.shape[0]
        exp_scores = np.exp(self.logit)  # compute exp logits
        grad = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + EPS)  # compute probabilities
        grad[range(num_train), y] -= 1
        grad = grad / num_train
        return grad

        ############################################################################
