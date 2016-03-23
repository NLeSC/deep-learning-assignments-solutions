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
    subgrad = np.zeros(W.shape)
    nrMarginOverflows = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        subgrad[:,j] = X[i]
        nrMarginOverflows += 1
    subgrad[:,y[i]] = -1*nrMarginOverflows*X[i]
    dW += subgrad

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  #Scale the gradient
  dW /= num_train


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_score = scores[xrange(num_train), y]
  margins = np.maximum(scores - np.transpose([correct_class_score]*num_classes) + 1, 0)
  margins[xrange(num_train), y] = 0
  loss = np.sum(margins)/num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #Compute the gradient
  margindices_incorrect = 1*(margins > 0)
  nOverflow = margindices_incorrect.sum(1)
  true_indices = np.zeros(margindices_incorrect.shape)
  true_indices[xrange(num_train),y] = 1
  margindices_correct = true_indices * np.transpose([nOverflow]*num_classes)
  
  dW = X.transpose().dot(margindices_incorrect) - X.transpose().dot(margindices_correct) 
  dW /= num_train

  return loss, dW
