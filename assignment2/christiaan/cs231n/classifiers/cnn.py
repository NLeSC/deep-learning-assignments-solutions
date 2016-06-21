import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # Intialization from Dafne:
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size
    stride = 1
    pad = (filter_size - 1) / 2
    H2 = (1 + (H + 2 * pad - HH) / stride) / 2
    W2 = (1 + (W + 2 * pad - WW) / stride) / 2

    self.params['W1'] = np.random.normal(0, weight_scale,
                                         (num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (H2 * W2 * num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X1 = X
    X2, cache1 = conv_relu_pool_forward(X1, W1, b1, conv_param, pool_param)
    X3, cache2 = affine_relu_forward(X2, W2, b2)
    X4, cache3 = affine_forward(X3, W3, b3)
    scores = X4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    softmax_l, dout = softmax_loss(scores, y)
    reg_W1_loss, dW1_reg = self.regularization_loss(self.params['W1'])
    reg_W2_loss, dW2_reg = self.regularization_loss(self.params['W2'])
    reg_W3_loss, dW3_reg = self.regularization_loss(self.params['W3'])
    loss = softmax_l + reg_W1_loss + reg_W2_loss

    dx3, dW3_sm, db3 = affine_backward(dout, cache3)
    dx2, dW2_sm, db2 = affine_relu_backward(dx3, cache2)
    dx1, dW1_sm, db1 = conv_relu_pool_backward(dx2, cache1)

    grads['W1'] = dW1_reg + dW1_sm
    grads['W2'] = dW2_reg + dW2_sm
    grads['W3'] = dW3_reg + dW3_sm
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

  def regularization_loss(self, W):
    loss = self.reg * 0.5 * np.sum(W * W)
    dx = self.reg * W
    return loss, dx
  
pass
