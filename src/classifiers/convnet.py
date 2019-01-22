import numpy as np

from src.layers import *
from src.fast_layers import *
from src.layer_utils import *

class ConvNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities. We also have an option for Softmax or SVM loss function.
  The network can be built with the following architecture types:

  1: [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  2: [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  3: [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, hidden_dims, num_filters, input_dim=(3, 32, 32), filter_size=7,
               use_batchnorm = False, architecture_type = 1, loss_type = 'softmax',
               num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
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
    self.num_layers_conv = len(num_filters)
    self.num_layers_affine = 1+len(hidden_dims)
    self.use_batchnorm = use_batchnorm
    self.architecture_type = architecture_type
    self.loss_type = loss_type
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
    C, H, W = input_dim
    stride = 1
    pool_h = 2
    pool_w = 2
    pool_s = 2
    # cascades of conv-relu-pool or conv-relu-conv-relu-pool
    for i in range(self.num_layers_conv):
        W_name = 'W' + str(i+1)
        b_name = 'b' + str(i+1)
        fH = fW = filter_size
        if self.architecture_type == 3:
            if i % 2 == 0:
                self.params[W_name] = np.random.normal(mu, sigma, (num_filters[i], C, fH, fW))
                self.params[b_name] = np.zeros(num_filters[i])
        else:
            # for architectures of type1
            if self.architecture_type == 1:
                if i < self.num_layers_conv-1:
                    self.params[W_name] = np.random.normal(mu, sigma, (num_filters[i], C, fH, fW))
                    self.params[b_name] = np.zeros(num_filters[i])
            else:
                # for architectures of type 2
                self.params[W_name] = np.random.normal(mu, sigma, (num_filters[i], C, fH, fW))
                self.params[b_name] = np.zeros(num_filters[i])
        h_out1 = (H + 2*(fH-1)/2 - fH)/stride + 1
        w_out1 = (W + 2*(fW-1)/2 - fW)/stride + 1
        if self.architecture_type == 3:
            if i % 2 == 0:
                C, H, W = num_filters[i], h_out1, w_out1
                # naming trick
                W_name = 'W' + str(i+2)
                b_name = 'b' + str(i+2)
                fH = fW = filter_size
                self.params[W_name] = np.random.normal(mu, sigma, (num_filters[i+1], C, fH, fW))
                self.params[b_name] = np.zeros(num_filters[i+1])
                h_out1 = (H + 2*(fH-1)/2 - fH)/stride + 1
                w_out1 = (W + 2*(fW-1)/2 - fW)/stride + 1
        # pooling layer for all architectures
        h_in2 = (h_out1-pool_h)/pool_s + 1
        w_in2 = (w_out1-pool_w)/pool_s + 1
        if self.architecture_type == 3:
            if i % 2 == 0 :
                C, H, W = num_filters[i+1], h_in2, w_in2
        elif self.architecture_type == 1:
            if i < self.num_layers_conv-1:
                C, H, W = num_filters[i], h_in2, w_in2
        else:
            C, H, W = num_filters[i], h_in2, w_in2
    
    # extra conv-relu layer for architecture type == 1
    if self.architecture_type == 1:
        i = i
        W_name = 'W' + str(i+1)
        b_name = 'b' + str(i+1)
        fH = fW = filter_size
        self.params[W_name] = np.random.normal(mu, sigma, (num_filters[i], C, fH, fW))
        self.params[b_name] = np.zeros(num_filters[i])
        h_in2 = (H + 2*(fH-1)/2 - fH)/stride + 1
        w_in2 = (W + 2*(fW-1)/2 - fW)/stride + 1
        C, H, W = num_filters[i], h_in2, w_in2
    
    # the affines
    input_dim_affine = C*H*W
    for j in range(self.num_layers_affine):
        W_name = 'W' + str(i+j+2)
        b_name = 'b' + str(i+j+2)
        if(j == 0):
            self.params[W_name] = np.random.normal(mu, sigma, (input_dim_affine, hidden_dims[j]))
            self.params[b_name] = np.zeros(hidden_dims[j])
        elif(j == self.num_layers_affine-1):
            self.params[W_name] = np.random.normal(mu, sigma, (hidden_dims[j-1], num_classes))
            self.params[b_name] = np.zeros(num_classes)
        else:
            self.params[W_name] = np.random.normal(mu, sigma, (hidden_dims[j-1], hidden_dims[j]))
            self.params[b_name] = np.zeros(hidden_dims[j])
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = {}
    # batch norm is being implemented here
    if self.use_batchnorm:
        for ic in range(self.num_layers_conv):
            if self.architecture_type != 3:
                bn_name = 'bn_name' + str(ic)
                gamma_name = 'gamma' + str(ic)
                beta_name = 'beta' + str(ic)
                self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                           np.zeros(num_filters[ic]), 'running_var':\
                                           np.zeros(num_filters[ic])}
                self.params[gamma_name] = np.ones(num_filters[ic])
                self.params[beta_name] = np.zeros(num_filters[ic])
                
            else:
                if ic % 2 == 0:
                    bn_name = 'bn_name' + str(ic)
                    gamma_name = 'gamma' + str(ic)
                    beta_name = 'beta' + str(ic)
                    self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                               np.zeros(num_filters[ic]), 'running_var':\
                                               np.zeros(num_filters[ic])}
                    self.params[gamma_name] = np.ones(num_filters[ic])
                    self.params[beta_name] = np.zeros(num_filters[ic])
                    bn_name = 'bn_name' + str(ic+1)
                    gamma_name = 'gamma' + str(ic+1)
                    beta_name = 'beta' + str(ic+1)
                    self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                               np.zeros(num_filters[ic+1]), 'running_var':\
                                               np.zeros(num_filters[ic+1])}
                    self.params[gamma_name] = np.ones(num_filters[ic+1])
                    self.params[beta_name] = np.zeros(num_filters[ic+1])
            
        for ia in range(self.num_layers_affine-1):
            bn_name = 'bn_name' + str(ic+ia+1)
            gamma_name = 'gamma' + str(ic+ia+1)
            beta_name = 'beta' + str(ic+ia+1)
            self.bn_params[bn_name] = {'mode' : 'train', 'running_mean': \
                                       np.zeros(hidden_dims[ia]), 'running_var':\
                                       np.zeros(hidden_dims[ia])}
            self.params[gamma_name] = np.ones(hidden_dims[ia])
            self.params[beta_name] = np.zeros(hidden_dims[ia])
    
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

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache = {}
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # calculate the regularization also parallely
    # convolution layers first
    reg_loss = 0
    if self.architecture_type == 1:
        for i in range(self.num_layers_conv-1):
            # pass conv_param to the forward pass for the convolutional layer
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            W1 = self.params[W_name]
            b1 = self.params[b_name]
            filter_size = W1.shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # forward_pass
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            if self.use_batchnorm:
                bn_name = 'bn_name' + str(i)
                gamma_name = 'gamma' + str(i)
                beta_name = 'beta' + str(i)
                out1, cache[i+1] = conv_bnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param,\
                                                                self.params[gamma_name],\
                                                                self.params[beta_name],\
                                                                self.bn_params[bn_name])
            else:
                out1, cache[i+1] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            X = out1
        i = i+1 # keeping consistency
        W_name = 'W' + str(i+1)
        b_name = 'b' + str(i+1)
        W1 = self.params[W_name]
        b1 = self.params[b_name]
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        # forward pass
        reg_loss += 0.5*self.reg*np.sum(W1*W1)
        if self.use_batchnorm:
            bn_name = 'bn_name' + str(i)
            gamma_name = 'gamma' + str(i)
            beta_name = 'beta' + str(i)
            out1, cache[i+1] = conv_bnorm_relu_forward(X, W1, b1, conv_param,\
                                                       self.params[gamma_name], self.params[beta_name],\
                                                       self.bn_params[bn_name])
        else:
            out1, cache[i+1] = conv_relu_forward(X, W1, b1, conv_param)
    elif self.architecture_type == 2:
        for i in range(self.num_layers_conv):
            # pass conv_param to the forward pass for the convolutional layer
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            W1 = self.params[W_name]
            b1 = self.params[b_name]
            filter_size = W1.shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # forward_pass
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            if self.use_batchnorm:
                bn_name = 'bn_name' + str(i)
                gamma_name = 'gamma' + str(i)
                beta_name = 'beta' + str(i)
                out1, cache[i+1] = conv_bnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param,\
                                                                self.params[gamma_name],\
                                                                self.params[beta_name],\
                                                                self.bn_params[bn_name])
            else:
                out1, cache[i+1] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            X = out1
    else:
        for i in range(self.num_layers_conv/2):
            # pass conv_param to the forward pass for the convolutional layer
            W_name = 'W' + str(2*i+1)
            b_name = 'b' + str(2*i+1)
            W1 = self.params[W_name]
            b1 = self.params[b_name]
            filter_size = W1.shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # forward_pass
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            if self.use_batchnorm:
                bn_name = 'bn_name' + str(2*i)
                gamma_name = 'gamma' + str(2*i)
                beta_name = 'beta' + str(2*i)
                out1, cache[2*i+1] = conv_bnorm_relu_forward(X, W1, b1, conv_param,\
                                                            self.params[gamma_name],\
                                                            self.params[beta_name],\
                                                            self.bn_params[bn_name])
            else:
                out1, cache[i+1] = conv_relu_forward(X, W1, b1, conv_param)
            W_name = 'W' + str(2*i+2)
            b_name = 'b' + str(2*i+2)
            W1 = self.params[W_name]
            b1 = self.params[b_name]
            filter_size = W1.shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # forward_pass
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            if self.use_batchnorm:
                bn_name = 'bn_name' + str(2*i+1)
                gamma_name = 'gamma' + str(2*i+1)
                beta_name = 'beta' + str(2*i+1)
                out1, cache[2*i+2] = conv_bnorm_relu_pool_forward(out1, W1, b1, conv_param,\
                                                                    pool_param,\
                                                                    self.params[gamma_name],\
                                                                    self.params[beta_name],\
                                                                    self.bn_params[bn_name])
            else:
                out1, cache[2*i+2] = conv_relu_pool_forward(out1, W1, b1, conv_param, pool_param)
            X = out1
        # fixing the 2 hop error in i
        i = i*2+1
    # reshaping the output
    outX = np.reshape(out1, (out1.shape[0], out1.shape[1]*out1.shape[2]*out1.shape[3]))
    # affine layers
    for j in range(self.num_layers_affine):
        W_name = 'W' + str(i+j+2)
        b_name = 'b' + str(i+j+2)
        W1 = self.params[W_name]
        b1 = self.params[b_name]
        if(j < self.num_layers_affine-1):
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            if self.use_batchnorm:
                bn_name = 'bn_name' + str(i+j+1)
                gamma_name = 'gamma' + str(i+j+1)
                beta_name = 'beta' + str(i+j+1)
                out2, cache[i+j+2] = affine_bnorm_relu_forward(outX, W1, b1,\
                                                               self.params[gamma_name],\
                                                               self.params[beta_name], \
                                                               self.bn_params[bn_name])
            else:
                out2, cache[i+j+2] = affine_relu_forward(outX, W1, b1)
            outX = out2
        else:
            reg_loss += 0.5*self.reg*np.sum(W1*W1)
            out3, cache[i+j+2] = affine_forward(out2, W1, b1)
    # final scores
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
    if self.loss_type == 'softmax':
        data_loss, dloss = softmax_loss(scores, y)
    else:
        data_loss, dloss = svm_loss(scores, y)
    loss = data_loss + reg_loss
    
    # calculating gradients using chain rule
    # for affine layers
    # remember i = self.num_layers_conv-1
    aff_layer_nums = np.arange(self.num_layers_affine+i, i, -1)
    dx_af = dloss
    for k in aff_layer_nums:
        W_name = 'W' + str(k+1)
        b_name = 'b' + str(k+1)
        cache_name = k+1
        if k == aff_layer_nums[0]:
            dx_af, dw_af, db_af = affine_backward(dx_af, cache[cache_name])
        else:
            if self.use_batchnorm:
                gamma = 'gamma' + str(k)
                beta = 'beta' + str(k)
                dx_af, dw_af, db_af, grads[gamma], grads[beta] = \
                affine_bnorm_relu_backward(dx_af, cache[cache_name])
            else:
                dx_af, dw_af, db_af = affine_relu_backward(dx_af, cache[cache_name])
        grads[W_name] = dw_af + self.reg*self.params[W_name]
        grads[b_name] = db_af
    
    # for convolution layers
    # first reshape as per the last layer of the convolution
    dx_af1 = np.reshape(dx_af, (out1.shape[0],out1.shape[1],out1.shape[2],out1.shape[3]))
    dx_c = dx_af1
    # we have not updated variable 'i' till now and hence can use it
    conv_layer_nums = np.arange(i,-1,-1)
    if self.architecture_type == 1:
        for k in conv_layer_nums:
            W_name = 'W' + str(k+1)
            b_name = 'b' + str(k+1)
            cache_name = k+1
            if k == conv_layer_nums[0]:
                if self.use_batchnorm:
                    gamma = 'gamma' + str(k)
                    beta = 'beta' + str(k)
                    dx_c, dw_c, db_c, grads[gamma], grads[beta] = \
                    conv_bnorm_relu_backward(dx_c, cache[cache_name])
                else:
                    dx_c, dw_c, db_c = conv_relu_backward(dx_c, cache[cache_name])
            else:
                if self.use_batchnorm:
                    gamma = 'gamma' + str(k)
                    beta = 'beta' + str(k)
                    dx_c, dw_c, db_c, grads[gamma], grads[beta] = \
                    conv_bnorm_relu_pool_backward(dx_c, cache[cache_name])
                else:
                    dx_c, dw_c, db_c = conv_relu_pool_backward(dx_c, cache[cache_name])
            grads[W_name] = dw_c + self.reg*self.params[W_name]
            grads[b_name] = db_c
    elif self.architecture_type == 2:
        for k in conv_layer_nums:
            W_name = 'W' + str(k+1)
            b_name = 'b' + str(k+1)
            cache_name = k+1
            if self.use_batchnorm:
                gamma = 'gamma' + str(k)
                beta = 'beta' + str(k)
                dx_c, dw_c, db_c, grads[gamma], grads[beta] = \
                conv_bnorm_relu_pool_backward(dx_c, cache[cache_name])
            else:
                dx_c, dw_c, db_c = conv_relu_pool_backward(dx_c, cache[cache_name])
            grads[W_name] = dw_c + self.reg*self.params[W_name]
            grads[b_name] = db_c
    else:
        conv_layer_nums = np.arange(i/2,-1,-1)
        for k in conv_layer_nums:
            W_name = 'W' + str(2*k+2)
            b_name = 'b' + str(2*k+2)
            cache_name = 2*k+2
            if self.use_batchnorm:
                gamma = 'gamma' + str(2*k+1)
                beta = 'beta' + str(2*k+1)
                dx_c, dw_c, db_c, grads[gamma], grads[beta] = \
                conv_bnorm_relu_pool_backward(dx_c, cache[cache_name])
            else:
                dx_c, dw_c, db_c = conv_relu_pool_backward(dx_c, cache[cache_name])
            grads[W_name] = dw_c + self.reg*self.params[W_name]
            grads[b_name] = db_c
            W_name = 'W' + str(2*k+1)
            b_name = 'b' + str(2*k+1)
            cache_name = 2*k+1
            if self.use_batchnorm:
                gamma = 'gamma' + str(2*k)
                beta = 'beta' + str(2*k)
                dx_c, dw_c, db_c, grads[gamma], grads[beta] = conv_bnorm_relu_backward(dx_c, cache[cache_name])
            else:
                dx_c, dw_c, db_c = conv_relu_backward(dx_c, cache[cache_name])
            grads[W_name] = dw_c + self.reg*self.params[W_name]
            grads[b_name] = db_c
            
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass

