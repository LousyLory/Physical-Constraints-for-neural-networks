from copy import deepcopy
import numpy as np
from required_funcs import *

def canon_nets(model2, indices_to_copy_to, use_batchnorm = False):
    '''
    This function creates canonicalized networks
    indices_to_copy_to are the new orderings of the correspoing
    columns in W1 and rows in W2
    model2 is the network to be modified
    use_batchnorm is the flag to identify the use of batchnormalization
    '''
    model2p = deepcopy(model2)

    # altering networks
    # reduce distance between layer 1
    model2p.params['W1'] = model2p.params['W1'][:,indices_to_copy_to]
    model2p.params['b1'] = model2p.params['b1'][indices_to_copy_to]
    if use_batchnorm == True:
        model2p.params['gamma0'] = model2p.params['gamma0'][indices_to_copy_to]
        model2p.params['beta0'] = model2p.params['beta0'][indices_to_copy_to]
        model2p.bn_params[0]['running_mean'] = model2p.bn_params[0]['running_mean'][indices_to_copy_to]
        model2p.bn_params[0]['running_var'] = model2p.bn_params[0]['running_var'][indices_to_copy_to]
    
    model2p.params['W2'] = model2p.params['W2'][indices_to_copy_to, :]
    model2p.params['b2'] = model2p.params['b2'][indices_to_copy_to]
    if use_batchnorm == True:
        model2p.params['gamma1'] = model2p.params['gamma1'][indices_to_copy_to]
        model2p.params['beta1'] = model2p.params['beta1'][indices_to_copy_to]
        model2p.bn_params[1]['running_mean'] = model2p.bn_params[1]['running_mean'][indices_to_copy_to]
        model2p.bn_params[1]['running_var'] = model2p.bn_params[1]['running_var'][indices_to_copy_to]
    
    model2p.params['W3'] = np.copy(model2.params['W3'])
    model2p.params['b3'] = np.copy(model2.params['b3'])
    
    return model2p

def full_canon_nets(model1, model2, method = 'Hungarian'):
    '''
    does all the way up canonicalization
    ''' 
    model1p = deepcopy(model1)
    model2p = deepcopy(model2)

    # altering nets
    if method == 'greedy':
        ids1 = match_vals(model1p, model2p, method='greedy', is_normed='False', layer='W1')
        ids1 = ids1.astype(int)
    else:
        ids1 = match_vals(model1p, model2p, method='Hungarian', is_normed='False', layer='W1')
        ids1 = ids1.astype(int)
    # altering layer 1
    model2p.params['W1'] = model2p.params['W1'][:, ids1]
    model2p.params['b1'] = model2p.params['b1'][ids1]
    #altering layer 2
    model2p.params['W2'] = model2p.params['W2'][ids1, :]
    model2p.params['b2'] = np.copy(model2p.params['b2'])
    # altering layer 3
    model2p.params['W3'] = np.copy(model2.params['W3'])
    model2p.params['b3'] = np.copy(model2.params['b3'])

    # altering net layer 2
    if method == 'greedy':
        ids2 = match_vals(model1p, model2p, method='greedy', is_normed='False', layer='W2')
        ids2 = ids2.astype(int)
    else:
        ids2 = match_vals(model1p, model2p, method='Hungarian', is_normed='False', layer='W2')
        ids2 = ids2.astype(int)
    # altering layer 2
    model2p.params['W2'] = model2p.params['W2'][:, ids2]
    model2p.params['b2'] = model2p.params['b2'][ids2]
    # altering layer 3
    model2p.params['W3'] = model2p.params['W3'][ids2, :]
    model2p.params['b3'] = model2p.params['b3']

    return model2p