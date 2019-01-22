import numpy as np
from dist_fun import *
from model_utils import *

def match_vals(model1p, model2p, method = 'Hungarian', is_normed = 'False', layer= 'W1'):
    '''
    This method defines matching vectors on the basis of function
    '''
    if layer == 'W1':
        if is_normed == 'True':
            model2p_W1_alt = weight_normer(np.copy(model2p.params['W1']))
            model1p_W1_alt = weight_normer(np.copy(model1p.params['W1']))
            matrix = compute_distances_no_loops(model2p_W1_alt.T, model1p_W1_alt.T)
        elif is_normed == 'Erik':
            model1p = model_normer(model1p)
            model2p = model_normer(model2p)
            matrix = compute_distances_no_loops(model2p.params['W1'].T, model1p.params['W1'].T)
        else:
            matrix = compute_distances_no_loops(model2p.params['W1'].T, model1p.params['W1'].T)
        try:
            if method == 'Hungarian':
                from munkres import Munkres
                m = Munkres()
                indices = m.compute(matrix)
                indices_assignments = np.copy(indices)
                indices_to_copy_to = np.copy(indices_assignments[:,1])
                feat = np.copy(indices_to_copy_to)
            
        except ValueError:
            print("run from command line: pip install munkres")
    
        if method == 'greedy':
            indices = np.argsort(matrix, axis=0)
            used = np.array(range(indices.shape[0]))
            feat = np.zeros(indices.shape[0])
            for i in range(indices.shape[0]):
                for j in range(indices.shape[0]):
                    if indices[i,j] in used:
                        feat[i] = indices[i,j]
                        used[indices[i,j]] = 100
                        break
        if is_normed == 'Erik':
            return feat, model1p, model2p
        else:
            return feat
    elif layer == 'W2':
        if is_normed == 'True':
            model2p_W2_alt = weight_normer(np.copy(model2p.params['W2']))
            model1p_W2_alt = weight_normer(np.copy(model1p.params['W2']))
            matrix = compute_distances_no_loops(model2p_W2_alt.T, model1p_W2_alt.T)
        elif is_normed == 'Erik':
            model1p = model_normer(model1p)
            model2p = model_normer(model2p)
            matrix = compute_distances_no_loops(model2p.params['W2'].T, model1p.params['W2'].T)
        else:
            matrix = compute_distances_no_loops(model2p.params['W2'].T, model1p.params['W2'].T)
        try:
            if method == 'Hungarian':
                from munkres import Munkres
                m = Munkres()
                indices = m.compute(matrix)
                indices_assignments = np.copy(indices)
                indices_to_copy_to = np.copy(indices_assignments[:, 1])
                feat = np.copy(indices_to_copy_to)

        except ValueError:
            print("run from command line: pip install munkres")

        if method == 'greedy':
            indices = np.argsort(matrix, axis=0)
            used = np.array(range(indices.shape[0]))
            feat = np.zeros(indices.shape[0])
            for i in range(indices.shape[0]):
                for j in range(indices.shape[0]):
                    if indices[i, j] in used:
                        feat[i] = indices[i, j]
                        used[indices[i, j]] = 100
                        break
        if is_normed == 'Erik':
            return feat, model1p, model2p
        else:
            return feat