import numpy as np
from sklearn.neighbors import KDTree  # to compute distance

from . import helper
from . import GPU_TE as gte
from . import CPU_TE as cte

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

    if k < 1:
        raise AssertionError("K should be greater than or equal to 1")

    if embedding < 1:
        raise AssertionError(
            "The embedding must be greater than or equal to 1"
        )

    if type(X) is not np.ndarray:
        raise AssertionError("X should be a numpy array")

    if type(Y) is not np.ndarray:
        raise AssertionError("Y should be a numpy array")

    if X.ndim == 2 and X.shape[1] == 1:
        X = X[:, 0]
    elif X.ndim != 1:
        raise ValueError(
            "X must be one-dimensional or have shape (N, 1)"
        )

    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y[:, 0]
    elif Y.ndim != 1:
        raise ValueError(
            "Y must be one-dimensional or have shape (N, 1)"
        )

    if len(X) != len(Y):
        raise AssertionError("The length of X & Y are not equal")



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
