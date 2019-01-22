import numpy as np
import math
from collections import Counter

def compute_distances_no_loops(Y, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    sum1 = np.sum(np.power(X,2), axis=1)
    sum2 = np.sum(np.power(Y,2), axis=1)
    sum3 = 2*np.dot(X, Y.T)
    dists = sum1.reshape(-1,1) + sum2
    dists = np.sqrt(dists - sum3)
    
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
