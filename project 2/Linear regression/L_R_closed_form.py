def closed_form(X, Y, lambda_factor):
    I = np.identity(X.shape[1])
    theta = np.linalg.inv(X.T @ X + lambda_factor * I) @ X.T @ Y
    return theta
