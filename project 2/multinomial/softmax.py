def compute_probabilities(X, theta, temp_parameter):
    itemp = 1 / temp_parameter
    dot_products = itemp * theta.dot(X.T)
    max_of_columns = dot_products.max(axis=0)
    shifted_dot_products = dot_products - max_of_columns
    exponentiated = np.exp(shifted_dot_products)
    col_sums = exponentiated.sum(axis=0)
    return exponentiated / col_sums
