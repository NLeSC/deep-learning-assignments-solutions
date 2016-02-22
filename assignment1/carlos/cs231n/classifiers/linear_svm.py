import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
        loss += margin
        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg, delta=1):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  vdW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  vScores = X.dot(W)
  vCorrect_class_score = vScores[range(len(y)), y]
  vMargin = (vScores.T - vCorrect_class_score + 1).T
  vLoss = 0.0

  # if j == y[i] do not include in loss (or dW)
  mask = np.zeros(vMargin.shape)
  mask[range(mask.shape[0]),y] = 1

  vLoss = (vMargin-mask)[vMargin>0].sum()

  vdW = np.zeros(W.shape)
  i,j = np.nonzero(vMargin>0)
  for ii,jj in zip(i,j):
    vdW[:,y[ii]] -= X[ii,:]
    vdW[:,jj] += X[ii,:]

  idx = (j == y[i])
  vdWCorr = np.zeros(W.shape)# if j == y[i]
  for ii,jj in zip(i[idx],j[idx]):
    vdWCorr[:,y[ii]] += X[ii,:]
    vdWCorr[:,jj] -= X[ii,:]

  vdW -= vdWCorr

  loss = vLoss
  dW = vdW

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

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
