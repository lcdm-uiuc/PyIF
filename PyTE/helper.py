import numpy as np


def make_spaces(X, Y, embedding=1):
    '''
    function to make 4 subspaces

    Parameters
    ----------
    x: numpy array
    y: numpy array
    embedding: how far to look back when creatining spaces. Must be greater than or equal to 1.

    Returns
    -------
    xky subspace: contains the X values, the embedding for X, and the Y Values
    ky subspace: contains X values and Y Values
    xk subspace: contains X values, the embedding values for X
    k subspace: contains only X Values
    '''


    # A column for X & Y, along with the embedding
    dimxky =  embedding + 2

    # A column for embedding and Y Value
    dimky  =  embedding + 1

    # A column for X & embedding values
    dimxk  =  embedding + 1

    # columns for embedding values
    dimk   =  embedding



    N = len(X)

    # Create numpy array of zeros for subspace
    xky_pts = np.zeros((N-embedding, dimxky))
    ky_pts = np.zeros((N-embedding, dimky))
    xk_pts = np.zeros((N-embedding, dimxk))
    k_pts = np.zeros((N-embedding, dimk))

    
    # Set the last column to the Y column. Start at the embedding index and take up to N - embedding values
    xky_pts[:, embedding+1] = Y.flatten()[embedding-1:][0:N-embedding]
    ky_pts[:, embedding] = Y.flatten()[embedding-1:][0:N-embedding]


    # start from embedding value and decrease to 0
    for i, j in enumerate(range(embedding, -1, -1)):

        # set first column to the X values from embedding to the length of the array
        # then take from 0 to N-embedding
        xky_pts[:, i] = X.flatten()[j:][0:N-embedding]
        xk_pts[:, i] = X.flatten()[j:][0:N-embedding]

        # This is so that there are no values of X in k or the ky subspaces.
        if i > 0:
            k_pts[:, i-1] = X.flatten()[j:][0:N-embedding]
            ky_pts[:, i-1] = X.flatten()[j:][0:N-embedding]

        # Repeat with next column from the embedding -1 to the length of the array
        # then take 0 to N-embedding
    return xky_pts, ky_pts, xk_pts, k_pts, N-embedding


def safetyCheck(X,Y):
    '''
    Checks for duplicate data and ends TE estimation if duplicate
    data points are found

    Parameters
    ----------
    X: Array that holds the X values
    Y: Array that holds the Y values

    Returns
    -------
    True if the safety check passes and False otherwise
    '''
    checkDict = {}
    for i in range(len(X)):
        if checkDict.get((X[i],Y[i])) == None:
            checkDict[(X[i],Y[i])] = 1
        else:
            return False
    return True

