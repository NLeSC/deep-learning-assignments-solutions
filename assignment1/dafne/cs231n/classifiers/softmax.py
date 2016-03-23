import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    maxscore = np.max(scores)
    shifted_scores = scores - maxscore
    probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores))
    loss += -np.log( probs[y[i]] )
    for j in xrange(num_classes):
        if y[i] == j:
            dW[:,j] += -X[i] * (1-probs[j])
        else:
             dW[:,j] += -X[i] * (-probs[j])
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  maxscore = np.max(scores, 1)
  shifted_scores = scores - maxscore[:,np.newaxis]
  probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), 1)[:,np.newaxis]
  
  losses = -np.log( probs[xrange(num_train), y] )
  loss = losses.sum() / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  true_indices = np.zeros(probs.shape)
  true_indices[xrange(num_train),y] = 1
  dW = - X.transpose().dot(true_indices - probs)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

