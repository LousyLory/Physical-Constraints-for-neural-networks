import numpy as np

from src.layers import *
from src.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    mu, sigma = 0, weight_scale  # mean and standard deviation
    self.params['W1'] = np.random.normal(mu, sigma, (input_dim, hidden_dim))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(mu, sigma, (hidden_dim, num_classes))
    self.params['b2'] = np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    reg = self.reg
    
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    # output of first layer
    out1, cache1 = affine_forward(X, W1, b1)
    # output of second layer
    out2, cache2 = relu_forward(out1)
    # output of third layer
    out3, cache3 = affine_forward(out2, W2, b2)
    scores = out3
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # computing the loss
    logC = np.max(out3, axis = 1)
    out3hat = (out3.T+logC).T
    row_sums1 = np.sum(np.exp(out3hat), axis = 1)
    row_sums = np.log(row_sums1)
    out4 = np.sum(-out3hat[np.arange(out2.shape[0]), y] + row_sums) / out2.shape[0]
    loss += out4 + 0.5 * reg * np.sum(W1 * W1)
    loss += 0.5 * reg * np.sum(W2 * W2)
    #loss += 0.5 * reg * np.sum(b1 * b1)
    #loss += 0.5 * reg * np.sum(b2 * b2)
    
    # computing grads
    dout5 = 1
    
    # gradient of softmax
    fraction_part = np.exp(out3hat)
    row_sums1 = row_sums1.reshape(row_sums1.shape[0],1)
    fraction_part /= row_sums1
    fraction_part[np.arange(out2.shape[0]), y] -= 1
    dout4 = fraction_part / X.shape[0]
    # chain rule
    dout4 *= dout5
    
    # chain rule
    doutX3, doutW3, doutb3 = affine_backward(dout4, cache3)
    grads['W2'] = doutW3 + reg*W2
    
    # chain rule for b2
    #grads['b2'] = doutb3 + reg*b2.T
    grads['b2'] = doutb3
    
    # chain rule for w1
    dout2 = relu_backward(doutX3, cache2)
    _, doutW1, doutb1 = affine_backward(dout2, cache1)
    grads['W1'] = doutW1 + reg*W1
    
    # chain rule for b1
    #grads['b1'] = doutb1 + reg*b1.T
    grads['b1'] = doutb1
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None, mixing_param=0.0):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    # process for spatial constrains
    cx, cy = np.mgrid[-np.sqrt(hidden_dims[0])/2:np.sqrt(hidden_dims[0])/2:1, \
                        -np.sqrt(hidden_dims[0])/2:np.sqrt(hidden_dims[0])/2:1]
    L1_dist_mat = cx+cy
    self.L1_arr = np.asarray(L1_dist_mat).reshape(-1)
    cx2, cy2 = np.power(cx,2), np.power(cy,2)
    L2_dist_mat = np.sqrt(cx2+cy2)
    self.L2_arr = np.asarray(L2_dist_mat).reshape(-1)
    self.mixing_param = mixing_param

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    mu, sigma = 0, weight_scale
    for i in range(self.num_layers):
        W_name = 'W' + str(i+1)
        b_name = 'b' + str(i+1)
        if(i == 0):
            self.params[W_name] = np.random.normal(mu, sigma, (input_dim, hidden_dims[i]))
            self.params[b_name] = np.zeros(hidden_dims[i])
        elif(i == self.num_layers-1):
            self.params[W_name] = np.random.normal(mu, sigma, (hidden_dims[i-1], num_classes))
            self.params[b_name] = np.zeros(num_classes)
        else:
            self.params[W_name] = np.random.normal(mu, sigma, (hidden_dims[i-1], hidden_dims[i]))
            self.params[b_name] = np.zeros(hidden_dims[i])
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      print 'We use dropout with p =%f' % (self.dropout_param['p'])
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = {}
    if self.use_batchnorm:
      for i in range(self.num_layers-1):
        bn_name = i
        gamma_name = 'gamma' + str(i)
        beta_name = 'beta' + str(i)
        self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                   np.zeros(hidden_dims[i]), 'running_var': np.zeros(hidden_dims[i])}
        self.params[gamma_name] = np.ones(hidden_dims[i])
        self.params[beta_name] = np.zeros(hidden_dims[i])
    #'''
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    out2 = np.copy(X)
    reg_loss = 0
    cache = {}
    for i in range(self.num_layers):
        x = np.copy(out2)
        if(i < self.num_layers-1):
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            CA_name = 'CA' + str(i+1)
            CR_name = 'CR' + str(i+1)
            BN_name = i
            g_name = 'gamma' + str(i)
            beta_name = 'beta' + str(i)
            DP_name = 'DP' + str(i)
            if self.use_dropout:
                x, cache[DP_name] = dropout_forward(x, self.dropout_param)
            out2, cache[CA_name] = affine_forward(x, self.params[W_name], self.params[b_name])
            ##################################################################################
            #  adding spatial constraint 
            if W_name == 'W1':
                #out2 = (1-self.mixing_param)*out2 + self.mixing_param*self.L1_arr
                out2 += self.mixing_param*self.L1_arr
                pass
            ##################################################################################
            reg_loss += 0.5*self.reg*np.sum(self.params[W_name]*self.params[W_name])
            if self.use_batchnorm:
                out2, cache[BN_name] = batchnorm_forward(out2, self.params[g_name], \
                                       self.params[beta_name], self.bn_params[BN_name])
            out2, cache[CR_name] = relu_forward(out2)
        else:
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            CA_name = 'CA' + str(i+1)
            DP_name = 'DP' + str(i)
            if self.use_dropout:
                x, cache[DP_name] = dropout_forward(x, self.dropout_param)
            out2, cache[CA_name] = affine_forward(x, self.params[W_name], self.params[b_name])
            reg_loss += 0.5*self.reg*np.sum(self.params[W_name]*self.params[W_name])
        out1 = np.copy(x)
        
    scores = out2
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # computing the loss
    logC = np.max(scores, axis = 1)
    out3hat = (scores.T+logC).T
    row_sums1 = np.sum(np.exp(out3hat), axis = 1)
    row_sums = np.log(row_sums1)
    out4 = np.sum(-out3hat[np.arange(out1.shape[0]), y] + row_sums) / out1.shape[0]
    loss += out4 + reg_loss
    '''
    if np.isnan(loss):
        if np.isnan(scores).any():
            print "scores: ", scores
        return 0
    '''
    '''
    for i in range(self.num_layers):
        W_name = 'W'+str(i+1)
        loss += 0.5*self.reg*np.sum(self.params[W_name]*self.params[W_name])
    '''
    # computing grads
    dout5 = 1
    # gradient of softmax
    fraction_part = np.exp(out3hat)
    row_sums1 = row_sums1.reshape(row_sums1.shape[0],1)
    fraction_part /= row_sums1
    fraction_part[np.arange(out1.shape[0]), y] -= 1
    dout4 = fraction_part / X.shape[0]
    # chain rule
    dout4 *= dout5
    
    arr = np.arange(self.num_layers,0,-1)
    
    for i in arr:
        W_name = 'W'+str(i)
        b_name = 'b'+str(i)
        CA_name = 'CA' + str(i)
        CR_name = 'CR' + str(i)
        BN_name = i-1
        g_name = 'gamma' + str(i-1)
        beta_name = 'beta' + str(i-1)
        DP_name = 'DP' + str(i-1)
        if(i == arr[0]):
            # chain rule for final update
            doutX3, doutW3, doutb3 = affine_backward(dout4, cache[CA_name])
            if self.use_dropout:
                doutX3 = dropout_backward(doutX3, cache[DP_name])
            grads[W_name] = doutW3 + self.reg*self.params[W_name]
            grads[b_name] = doutb3
        else:
            dout2 = relu_backward(doutX3, cache[CR_name])
            if self.use_batchnorm :
                dout2, dgamma, dbeta = batchnorm_backward(dout2, cache[BN_name])
                grads[g_name] = dgamma
                grads[beta_name] = dbeta
            doutX3, doutW3, doutb3 = affine_backward(dout2, cache[CA_name])
            if self.use_dropout:
                doutX3 = dropout_backward(doutX3, cache[DP_name])
            grads[W_name] = doutW3 + self.reg*self.params[W_name]
            grads[b_name] = doutb3
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
