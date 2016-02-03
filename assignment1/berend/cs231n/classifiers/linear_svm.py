import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    h = 0.00001
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    delta_total = 0.0
    for i in xrange(num_train):
        scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        count_delta = 0.0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                count_delta += 1
                loss += margin
                dW[j] += X[:, i]
        delta_total += count_delta
        dW[y[i]] -= count_delta * X[:, i]

    print "Naive total delta: ", delta_total
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    #dW += 0.5 * reg * np.sum(dW * dW)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_classes = W.shape[0]
    num_train = X.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = W.dot(X)
    correct_indices = np.array([y, range(num_train)])
    correct_class_score = scores[correct_indices[0], correct_indices[1]]
    margins = np.maximum(0, scores - correct_class_score + 1.0)  # note delta = 1

    margins[correct_indices[0], correct_indices[1]] = 0

    loss = np.sum(margins)
    count_delta = np.sum(margins > 0)
#    loss_row_sum = np.zeros(X.shape[0])
#    for j in xrange(num_classes):
#            if j == y[i]:
#                continue
#            loss += np.sum(margins[margins > 0])
#            count_delta += np.sum(margins > 0)
#            row_sum = np.sum(X[:, margins > 0], axis=1);
#            loss_row_sum += row_sum
#            dW[j] += row_sum
#    dW[y[i]] -= count_delta * loss_row_sum

    print "Vectorized total delta: ", count_delta
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
