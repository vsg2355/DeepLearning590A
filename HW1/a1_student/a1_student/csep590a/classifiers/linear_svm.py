from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    dW = dW / num_train + reg * 2 * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_train, D = X.shape
    scores = X.dot(W)  # (N, C)
    range_num_train = list(range(num_train))
    correct_class_scores = scores[range_num_train, y][:, np.newaxis]
    margins = np.maximum(0, scores - correct_class_scores + 1.0)
    margins[range_num_train, y] = 0

    loss_cost = np.sum(margins) / num_train
    loss_reg = reg * np.sum(W * W)
    loss = loss_cost + loss_reg

    num_pos = np.sum(margins > 0, axis=1)  # number of positive losses

    dscores = np.zeros(scores.shape)
    dscores[margins > 0] = 1
    dscores[range_num_train, y] = -num_pos

    dW = X.T.dot(dscores) / num_train + reg * 2 * W

    return loss, dW
