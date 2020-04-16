import numpy as np
eps=1e-6

def whiten(X, reg):
    """
    Whiten the input feature matrix X, which is of dimension NxD.
    A regularization constant (reg>0) is added to the diagonal of covariance to ensure numerical stability, which also
    acts as soft shrinkage.
    The value of reg relies on proper scaling of X (e.g., the variance of each feature dimension).
    """
    N, D = X.shape
    cov = np.matmul(X.transpose(), X) / (N-1)
    cov = cov + reg * np.eye(D, D)
    # Extract eigen-system.
    E, V = np.linalg.eig(cov)
    # Whitening transform.
    S = np.zeros_like(E)
    S[np.where(E>eps)] = np.divide(1.0, np.sqrt(E[np.where(E>eps)]))
    W = np.matmul(np.multiply(V, S[np.newaxis,:]), V.transpose())
    
    # Return the whitened X, and the whitening transformation.
    return np.matmul(X, W), W

def MCCA(K, dim, Xs, rs):
    """
    A straightforward implementation of multi-view CCA.
    
    inputs:
    	k: number of views
        dim: dimension of projection
        Xs: list of feature matrices, each of dimension NxD_i
        rs: list of regularization parameters
    outputs:
        CORR: array of correlations in projected dimensions
    	UU: list of projection
    	MM: list of mean vectors
        X_PROJ: projected datasets
    """

    # Remove the mean, and whiten the data.
    DD = [0]
    MM = []
    Y = []
    W = []
    for i in range(K):
        Xi = Xs[i]
        DD.append(Xi.shape[1])
        m = np.mean(Xi, axis=0, keepdims=True)
        Y_tmp, W_tmp = whiten(np.subtract(Xi, m), rs[i])
        MM.append(m)
        Y.append(Y_tmp)
        W.append(W_tmp)

    # Construct the big eigen-system.
    OFFSET = np.cumsum(DD)
    Y_tmp = np.concatenate(Y, axis=1)
    A = np.matmul(Y_tmp.transpose(), Y_tmp) / (Y_tmp.shape[0]-1)
    CORR, PROJ = np.linalg.eig(A)
    # Sort in decreasing order.
    IDX = np.argsort(CORR)
    CORR = CORR[IDX[::-1]]
    PROJ = PROJ[:, IDX[::-1]]

    # Retrieve the block of projection matrix for each view, combined with the whitening transform.
    X_PROJ = []
    UU = []
    for i in range(K):
        row_idx1 = OFFSET[i]
        row_idx2 = OFFSET[i+1]
        # Here we do use a heuristic for scaling PROJ that is consistent with the two view case.
        # In such a way, in the two view case, the projected data with will unit scale in each dimension.
        PROJ_tmp = PROJ[row_idx1:row_idx2, :dim]
        scale = np.sqrt(np.sum(np.square(PROJ_tmp), axis=0, keepdims=True))
        PROJ_tmp = np.divide(PROJ_tmp, scale)
        UU.append(np.matmul(W[i], PROJ_tmp))
        X_PROJ.append(np.matmul(Y[i], PROJ_tmp))

    # Return the canonical correlations, projection matrices, and projected data.
    return CORR, UU, MM, X_PROJ


        

    

    
