def polynomial_kernel(X, Y, c, p):
    K = X @ Y.transpose()
    K += c
    K **= p
    return K
