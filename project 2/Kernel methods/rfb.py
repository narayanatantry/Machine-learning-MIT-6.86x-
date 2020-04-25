def rbf_kernel(X, Y, gamma):
    
    XTX = np.mat([np.dot(row, row) for row in X]).T
    YTY = np.mat([np.dot(row, row) for row in Y]).T
    XTX_matrix = np.repeat(XTX, Y.shape[0], axis=1)
    YTY_matrix = np.repeat(YTY, X.shape[0], axis=1).T
    K = np.asarray((XTX_matrix + YTY_matrix - 2 * (X @ Y.T)), dtype='float64')
    K *= - gamma
    return np.exp(K, K)
