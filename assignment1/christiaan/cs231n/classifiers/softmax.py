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

  num_images = X.shape[0]

  print 'num_images', num_images

  num_classes = W.shape[1]

  print 'num_classes', num_classes


  for im in range(0, num_images):
    dW_im = np.zeros_like(dW)
    scores_im = W.T.dot(X[im,:])
    log_c = -np.max(scores_im) # try 0
    correct_class = y[im]
    exp_correct_class = np.exp(scores_im[correct_class] + log_c)

    sum_exp_classes = 0
    for cl in range(0, num_classes):
      sum_exp_classes += np.exp(scores_im[cl] + log_c)

    for cl in range(0, num_classes):
      exp_current_class = np.exp(scores_im[cl] + log_c)
      softmax_current_class = exp_current_class / sum_exp_classes
      if cl == correct_class:
        dW_im[:,cl] = (-1 + softmax_current_class) * X[im,:]
      else:
        dW_im[:,cl] = softmax_current_class * X[im,:]
    softmax_correct_class = exp_correct_class / sum_exp_classes
    loss_im = -np.log(softmax_correct_class)
    loss += loss_im

    dW += dW_im

  # ### again, without numerical stability trick (or gradient)
  # loss = 0.0
  # for im in range(0, num_images):
  #   summation = 0
  #   scores_im = W.T.dot(X[im,:])
  #   assert scores_im.shape[0] == num_classes
  #   for cl in range(0, num_classes):
  #     summation += np.exp(scores_im[cl])
  #   log_im = np.log(summation)
  #   loss_im = log_im - scores_im[y[im]]
  #   loss += loss_im
  # ### end again


  # Convert sum over images to mean.
  loss /= num_images
  dW /= num_images

  # regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W




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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

