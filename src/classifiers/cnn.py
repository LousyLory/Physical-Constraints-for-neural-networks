import numpy as np

from src.layers import *
from src.fast_layers import *
from src.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, use_batchnorm = False, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
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
    self.use_batchnorm = use_batchnorm
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
    mu, sigma = 0, weight_scale
    # first layer
    C, H, W = input_dim
    fH = fW = filter_size
    stride = 1
    self.params['W1'] = np.random.normal(mu, sigma, (num_filters, C, fH, fW))
    self.params['b1'] = np.zeros(num_filters)
    h_out1 = (H + 2*(fH-1)/2 - fH)/stride + 1
    w_out1 = (W + 2*(fW-1)/2 - fW)/stride + 1
    # pooling layer
    pool_h = 2
    pool_w = 2
    pool_s = 2
    h_in2 = (h_out1-pool_h)/pool_s + 1
    w_in2 = (w_out1-pool_w)/pool_s + 1
    # hidden affine
    self.params['W2'] = np.random.normal(mu, sigma, (num_filters*h_in2*w_in2, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    # penultimate layer
    self.params['W3'] = np.random.normal(mu, sigma, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = {}
    if self.use_batchnorm:
        bn_name = 0
        gamma_name = 'gamma' + str(0)
        beta_name = 'beta' + str(0)
        self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                   np.zeros(num_filters), 'running_var': np.zeros(num_filters)}
        self.params[gamma_name] = np.ones(num_filters)
        self.params[beta_name] = np.zeros(num_filters)
        bn_name = 1
        gamma_name = 'gamma' + str(1)
        beta_name = 'beta' + str(1)
        self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                   np.zeros(hidden_dim), 'running_var': np.zeros(hidden_dim)}
        self.params[gamma_name] = np.ones(hidden_dim)
        self.params[beta_name] = np.zeros(hidden_dim)
        
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    
    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode
    
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
    # output of first conv layer
    if self.use_batchnorm:
        out1, cache1 = conv_bnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param,\
                                                    self.params['gamma0'], self.params['beta0'],\
                                                    self.bn_params[0])
    else:
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # reshaping the output
    outX = np.reshape(out1, (out1.shape[0], out1.shape[1]*out1.shape[2]*out1.shape[3]))
    # output of second affine layer
    if self.use_batchnorm:
        out2, cache2 = affine_bnorm_relu_forward(outX, W2, b2, self.params['gamma1'],\
                                                 self.params['beta1'], self.bn_params[1])
    else:
        out2, cache2 = affine_relu_forward(outX, W2, b2)
    # output of third layer
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3
    pass
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
    # calculating loss
    data_loss, dsoftmax = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W3 * W3)
    reg_loss += 0.5 * self.reg * np.sum(W2 * W2)
    reg_loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss = data_loss + reg_loss
    # calculating gradients
    
    # chain rules
    dx_af2, dw_af2, db_af2 = affine_backward(dsoftmax, cache3)
    if self.use_batchnorm:
        dx_af1, dw_af1, db_af1, dgamma_af1, dbeta_af1 = affine_bnorm_relu_backward(dx_af2, cache2)
    else:
        dx_af1, dw_af1, db_af1 = affine_relu_backward(dx_af2, cache2)
    # reshape for conv layers
    dx_af1 = np.reshape(dx_af1, (out1.shape[0],out1.shape[1],out1.shape[2],out1.shape[3]))
    if self.use_batchnorm:
        dx_c, dw_c, db_c, dgamma_c, dbeta_c = conv_bnorm_relu_pool_backward(dx_af1, cache1)
    else:
        dx_c, dw_c, db_c = conv_relu_pool_backward(dx_af1, cache1)
    
    grads['W1'] = dw_c + self.reg*W1
    grads['b1'] = db_c
    grads['W2'] = dw_af1 + self.reg*W2
    grads['b2'] = db_af1
    grads['W3'] = dw_af2 + self.reg*W3
    grads['b3'] = db_af2
    if self.use_batchnorm:
        grads['gamma0'] = dgamma_c
        grads['beta0'] = dbeta_c
        grads['gamma1'] = dgamma_af1
        grads['beta1'] = dbeta_af1
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

