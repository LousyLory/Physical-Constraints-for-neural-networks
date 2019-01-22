from random import randrange
import numpy as np

def check_fun(data2, model2, model2_alt, use_batchnorm = False):
    
    # checking section
    #model2 is the original one
    #model2_alt is the modified one
    m, n, o, p = data2['X_val'].shape
    inx = randrange(m-1)

    dummy_x = np.copy(data2['X_val'][inx,:])
    dummy_y = np.copy(data2['y_val'][inx])
    dummy_x = np.reshape(dummy_x,-1)
    # first forward pass
    xt_1 = np.dot(dummy_x, model2.params['W1']) + model2.params['b1']
    xt_2 = np.dot(dummy_x, model2_alt.params['W1']) + model2_alt.params['b1']
    
    # batch normalization
    if use_batchnorm == True:
        xt_1 = xt_1 - model2.bn_params[0]['running_mean']
        xt_1 = xt_1 / np.sqrt(model2.bn_params[0]['running_var']+1e-5)
        xt_1 = model2.params['gamma0']*xt_1 + model2.params['beta0']
    
        xt_2 = xt_2 - model2_alt.bn_params[0]['running_mean']
        xt_2 = xt_2 / np.sqrt(model2_alt.bn_params[0]['running_var']+1e-5)
        xt_2 = model2_alt.params['gamma0']*xt_2 + model2_alt.params['beta0']
    #'''
    print 'Test accuracy after one swap: ', np.linalg.norm(xt_1 - xt_2)

    # second forward pass
    xt1_1 = np.dot(xt_1, model2.params['W2']) + model2.params['b2']
    xt1_2 = np.dot(xt_2, model2_alt.params['W2']) + model2_alt.params['b2']

    # batch normalization
    if use_batchnorm == True:
        xt1_1 = xt1_1 - model2.bn_params[1]['running_mean']
        xt1_1 = xt1_1 / np.sqrt(model2.bn_params[1]['running_var']+1e-5)
        xt1_1 = model2.params['gamma1']*xt1_1 + model2.params['beta1']
    
        xt1_2 = xt1_2 - model2_alt.bn_params[1]['running_mean']
        xt1_2 = xt1_2 / np.sqrt(model2_alt.bn_params[1]['running_var']+1e-5)
        xt1_2 = model2_alt.params['gamma1']*xt1_2 + model2_alt.params['beta1']
    #'''
    print 'Test accuracy after two swaps: ', np.linalg.norm(xt1_1 - xt1_2)

    # third forward pass
    xt2_1 = np.dot(xt1_1, model2.params['W3']) + model2.params['b3']
    xt2_2 = np.dot(xt1_2, model2_alt.params['W3']) + model2_alt.params['b3']

    print 'Test accuracy after final swaps: ', np.linalg.norm(xt2_1 - xt2_2)