from copy import deepcopy

def create_model(model1, model2, alpha, use_batchnorm=False):
    '''
    This function creates new models as a convex combination of two other models
    alpha is the controlling parameter
    '''
    model3 = deepcopy(model1)
    
    for k, v in model1.params.iteritems():
        #print '%s: ' % k, v.shape
        model3.params[k] = alpha * model1.params[k] + (1-alpha) * model2.params[k]
    if use_batchnorm:
        for k, v in model1.bn_params.iteritems():
            for p,q in v.iteriterms():
                if p != 'mode':
                    model3.bn_params[k][p][q] = alpha * model1.bn_params[k][p][q] + (1-alpha) * model2.bn_params[k][p][q]
                else:
                    model3.bn_params[k][p][q] = model1.bn_params[k][p][q]
    return model3
        