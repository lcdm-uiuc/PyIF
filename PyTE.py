import numpy as np
from sklearn.neighbors import KDTree  # to compute distance

import helper
import GPU_TE as gte
import CPU_TE as cte

from nose.tools import assert_true

def te_compute(X, Y, k=1, embedding=1, safetyCheck=False, GPU=False):
    '''
    Parameters
    ----------
    X: numpy array
    Y: numpy array
    k: number of nearest neighbors
    embedding: integer containing number of lag periods to consider
    safetyCheck: Boolean value when True, will check for unique values
                 and abort estimation if duplicate values are found
    GPU: Boolean value that when set to true will use CUDA compatiable GPUs

    Returns
    -------
    TE: Floating point value
    '''

    assert_true(k>=1, msg="K should be greater than or equal to 1")
    assert_true(embedding >= 1, msg='The embedding must be greater than or equal to 1')
    assert_true(type(X) == np.ndarray, msg='X should be a numpy array')
    assert_true(type(Y) == np.ndarray, msg='Y should be a numpy array')
    assert_true(len(X) == len(Y), msg='The length of X & Y are not equal')



    if safetyCheck and (not helper.safetyCheck(X,Y)):
        print("Safety check failed. There are duplicates in the data.")
        return None

    # Make Spaces
    xkyPts, kyPts, xkPts, kPts, nPts = helper.make_spaces(X, Y,
     embedding=embedding)

    # Make Trees
    xkykdTree = KDTree(xkyPts, metric="chebyshev")
    kykdTree = KDTree(kyPts, metric="chebyshev")
    xkkdTree = KDTree(xkPts, metric="chebyshev")
    kkdTree = KDTree(kPts, metric="chebyshev")

    if GPU:
        TE = gte.compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
        xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=embedding, k=k)

    else:
        TE = cte.compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
        xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=embedding, k=k)


    return TE
