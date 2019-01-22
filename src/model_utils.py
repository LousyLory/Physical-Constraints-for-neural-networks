from copy import deepcopy
import numpy as np

def weight_normer(W1):
    '''
    This function takes in a model and normalizes
    its weights, i.e., the norm of the individual
    weight vector are set to one explicitly.
    '''
    '''
    This model has three layers, thus three weight
    matrices. We need to find the norms for each 
    one of these.
    '''
    W1_norms = np.sum(np.abs(W1)**2,axis=-0)**(1./2)
    W1 /= W1_norms
    return W1

def model_normer(model):
    '''
    This function normalizes the layer weights
    and updates the network for consistency
    '''
    new_model = deepcopy(model)
    W1_norms = np.sum(np.abs(new_model.params['W1'])**2,axis=-0)**(1./2)
    new_model.params['W1'] /= W1_norms
    pass
    W2 = np.copy(new_model.params['W2']).T
    W2 *= W1_norms
    new_model.params['W2'] = W2.T
    pass
    W2_norms = np.sum(np.abs(new_model.params['W2']) ** 2, axis=-0) ** (1. / 2)
    new_model.params['W2'] /= W2_norms
    pass
    W3 = np.copy(new_model.params['W3']).T
    W3 *= W2_norms
    new_model.params['W3'] = W3.T
    pass
    return new_model